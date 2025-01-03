import os
import json
import torch
import torch.nn as nn
import numpy as np
from peft import PeftModel
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM


# A helper to fix shapes for A, B
def fix_lora_shapes(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray):
    """
    x_data shape => (batch, seq_len, hidden_dim).
    We ensure A => [hidden_dim, r], B => [r, out_dim].
    """
    in_dim = x_data.shape[-1]
    a0, a1 = A.shape
    # A => [in_dim, r]
    if a0 == in_dim:
        r = a1
    elif a1 == in_dim:
        A = A.transpose(0, 1)
        r = A.shape[1]
    else:
        raise ValueError(f"A shape {A.shape} doesn't match x_data last dim {in_dim}.")

    b0, b1 = B.shape
    if b0 == r:
        out_dim = b1
    elif b1 == r:
        B = B.transpose(0, 1)
        out_dim = B.shape[1]
    else:
        raise ValueError(f"B shape {B.shape} doesn't match rank={r} in any dimension.")
    return A, B, in_dim, r, out_dim


class LoraApplyOneRow(nn.Module):
    """
    Expects shape (1, batch*seq_len*hidden_dim).
    Internal forward => reshape to (batch, seq_len, hidden_dim).
    """

    def __init__(self, A, B, batch_size, seq_len, hidden_dim):
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, x_1d):
        # x_1d => shape (1, total_size)
        x_3d = x_1d.view(self.batch_size, self.seq_len, self.hidden_dim)
        out_3d = (x_3d @ self.A) @ self.B
        out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
        # Flatten output for demonstration
        out_2d = out_3d.view(1, -1)
        return out_2d


def make_lora_hook(mod_name, activation_map):
    """
    Creates a forward hook function for LoRA modules that stores activations.
    
    :param mod_name: Name of the module to hook
    :param activation_map: Dictionary to store activations
    :return: Hook function
    """
    def hook(mod, layer_inputs, layer_output):
        if not layer_inputs:
            return
        x = layer_inputs[0]  # shape: (batch, seq_len, hidden_dim)
        print(f"shape in hook ({mod_name}):", x.size())
        activation_map[mod_name] = x.detach().cpu().numpy()

    return hook


def register_lora_hooks(model, activation_map, submodule_key=None):
    """
    Recursively finds LoRA submodules and registers forward hooks.
    
    :param model: The model to register hooks on
    :param activation_map: Dictionary to store activations
    :param submodule_key: Optional key to filter submodules
    :return: True if any wte/wpe warnings were issued
    """
    issued_wte_warning = False
    
    for full_name, module in model.named_modules():
        # Check if this submodule has LoRA
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            # Skip embedding submodules
            if "wte" in full_name or "wpe" in full_name:
                if not issued_wte_warning:
                    print(f"WARNING: Found LoRA submodule '{full_name}' (wte/wpe). Skipping hooking embeddings.")
                    issued_wte_warning = True
                continue

            # If user wants to filter e.g. "attn.c_attn"
            if submodule_key and submodule_key not in full_name:
                continue

            print(f"Registering hook on LoRA submodule: {full_name}")
            module.register_forward_hook(make_lora_hook(full_name, activation_map))
    
    return issued_wte_warning


def export_lora_submodules(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    input_texts: list[str],
    output_dir: str = "lora_onnx_params",
    json_dir: str = "intermediate_activations",
    submodule_key: str = None,
) -> None:
    """
    1) Captures LoRA sub-layer inputs with shape (batch, seq_len, hidden_dim).
    2) Flattens them into shape (1, batch*seq_len*hidden_dim).
    3) Exports an ONNX submodule that expects (1, total_size).
       - Inside that submodule's forward pass, it reshapes back to (batch, seq_len, hidden_dim).
    4) Writes a JSON file containing a single row of floats ( shape => (1, total_size) ).

    :param model:         A LoRA-augmented (PEFT) model, in eval mode.
    :param tokenizer:     A tokenizer (from the same or compatible base model).
    :param input_texts:   A list of strings for batched input. e.g. ["Hello", "More text", ...]
    :param output_dir:    Where to save ONNX files.
    :param json_dir:      Where to save JSON files.
    :param submodule_key: If set (e.g. "attn.c_attn"), export only submodules containing that key.
    """
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # Ensure we can pad if GPT-2-like
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # We'll store each sub-layer input in a dict
    activation_map = {}

    # Register hooks
    register_lora_hooks(model, activation_map, submodule_key)

    # Tokenize the input text as a single batch
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    print("input_ids shape:", input_ids.shape)

    # Run forward pass
    with torch.no_grad():
        _ = model(input_ids)

    # If no sub-layer activations were captured
    if len(activation_map) == 0:
        print("No LoRA sub-layer activations captured. Possibly no triggers for these inputs.")
        return

    # For each submodule hooking
    for full_name, x_data in activation_map.items():
        # x_data shape => (batch, seq_len, hidden_dim)
        batch_size = x_data.shape[0]
        seq_len = x_data.shape[1]
        hidden_dim = x_data.shape[2]

        total_size = batch_size * seq_len * hidden_dim  # e.g. 3*4*768=9216
        # Flatten to => (1, total_size)
        one_row = x_data.reshape(1, total_size)

        # Look up the submodule
        submodule = dict(model.named_modules()).get(full_name, None)
        if submodule is None:
            print(f"Cannot find submodule {full_name} in the model dict, skipping.")
            continue

        # Extract A,B
        if hasattr(submodule.lora_A, "keys"):
            a_keys = list(submodule.lora_A.keys())
            if not a_keys:
                print(f"No keys in submodule.lora_A for {full_name}, skipping.")
                continue
            A_mod = submodule.lora_A[a_keys[0]]
        else:
            A_mod = submodule.lora_A

        if hasattr(submodule.lora_B, "keys"):
            b_keys = list(submodule.lora_B.keys())
            if not b_keys:
                print(f"No keys in submodule.lora_B for {full_name}, skipping.")
                continue
            B_mod = submodule.lora_B[b_keys[0]]
        else:
            B_mod = submodule.lora_B

        if not hasattr(A_mod, "weight"):
            print(f"No weight in lora_A for {full_name}, skipping.")
            continue
        if not hasattr(B_mod, "weight"):
            print(f"No weight in lora_B for {full_name}, skipping.")
            continue

        A_raw = A_mod.weight.detach().cpu().float()
        B_raw = B_mod.weight.detach().cpu().float()

        # fix shapes
        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_shapes(
                A_raw, B_raw, x_data
            )
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            continue

        # Build sub-module expecting => (1, total_size)
        lora_mod = LoraApplyOneRow(
            A_fixed, B_fixed, batch_size, seq_len, hidden_dim
        ).eval()

        # Save ONNX
        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        x_tensor = torch.from_numpy(one_row)
        import onnx
        from torch.onnx import TrainingMode

        try:
            torch.onnx.export(
                lora_mod,
                x_tensor,
                onnx_path,
                export_params=True,
                do_constant_folding=False,
                opset_version=11,
                input_names=["input_x"],
                output_names=["output"],
                training=TrainingMode.TRAINING,
                keep_initializers_as_inputs=False,
            )
        except Exception as e:
            print(f"Export error for {full_name}: {e}")
            continue

        # Save JSON => single row of shape (1, total_size)
        data_json = {"input_data": one_row.tolist()}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path}, shape => {one_row.shape}")


###########################################################
# Example usage in another script:
#
# from zklora_one_row import export_lora_submodules_one_row
# from generate_proofs_async import generate_proofs_async   # your proof function
# import asyncio
#
# base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# lora_model = PeftModel.from_pretrained(base_model, "some-lora-adapter")
# lora_model.eval()
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#
# texts = [
#     "Hello from LoRA",
#     "Another line of text for multi-batch",
#     "One more line"
# ]
#
# export_lora_submodules_one_row(
#     model=lora_model,
#     tokenizer=tokenizer,
#     input_texts=texts,
#     output_dir="lora_onnx_params",
#     json_dir="intermediate_activations",
#     submodule_key="attn.c_attn"  # or None to export all LoRA submodules
# )
#
# # Then run proof generation:
# # asyncio.run(generate_proofs_async(
# #     onnx_dir="lora_onnx_params",
# #     json_dir="intermediate_activations",
# #     output_dir="proof_artifacts"
# # ))
#
###########################################################
