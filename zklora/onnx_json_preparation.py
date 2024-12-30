import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode
from typing import Union, List
from peft import PeftModel
from transformers import PreTrainedTokenizer, AutoTokenizer


def export_lora_submodules_flattened_multi(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    input_text: Union[str, List[str]],
    output_dir: str = "lora_onnx_params",
    json_dir: str = "intermediate_activations",
    submodule_key: str = None
) -> None:
    """
    Similar to export_lora_submodules_flattened, but `input_text` can be a list of strings.
    We then do a *batched* forward pass on all those strings at once, capturing sub-layer inputs
    shape [batch_size, seq_len, hidden_dim].
    
    1) If input_text is a single string, we treat it as list[str] of length 1.
    2) We flatten shape => [batch_size, seq_len * hidden_dim].
    3) Exports submodules with an ONNX input of shape [batch_size, (seq_len * hidden_dim)].
    4) Writes a JSON with the same 2D shape, so EZKL can parse easily.
    """

    # Make sure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # If user passes a single string, wrap it in a list
    if isinstance(input_text, str):
        input_text = [input_text]

    # We'll store sub-layer inputs in a dict
    activation_map = {}
    issued_wte_warning = False

    # -- NEW PART: ensure pad token is set
    if tokenizer.pad_token is None:
        # Approach #1: use EOS as pad
        tokenizer.pad_token = tokenizer.eos_token
        # Approach #2 (alternative): add a real PAD token and resize model
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # model.resize_token_embeddings(len(tokenizer))

    def register_lora_hooks_recursive(model, activation_map):
        """
        Recursively finds LoRA submodules, skipping 'wte'/'wpe' if found,
        and register a forward hook capturing the sub-layer input.
        """
        for full_name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # If submodule is named wte or wpe => skip hooking
                if ("wte" in full_name or "wpe" in full_name):
                    nonlocal issued_wte_warning
                    if not issued_wte_warning:
                        print(f"WARNING: Found LoRA submodule '{full_name}' (wte/wpe). Skipping hooking embeddings.")
                        issued_wte_warning = True
                    continue

                print(f"Registering hook on LoRA submodule: {full_name}")
                def make_hook(mod_name):
                    def hook(mod, layer_inputs, layer_output):
                        if not layer_inputs:
                            return
                        x = layer_inputs[0]  # shape: [batch_size, seq_len, hidden_dim] or sometimes [batch*seq_len, hidden_dim]
                        print(f"shape in hook ({mod_name}):", x.size())
                        activation_map[mod_name] = x.detach().cpu().numpy()
                    return hook

                module.register_forward_hook(make_hook(full_name))

    # Register hooking
    register_lora_hooks_recursive(model, activation_map)

    # Tokenize the list of strings in a single batch
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        padding=True,      # so they all have same seq_len
        truncation=True
    )
    input_ids = inputs["input_ids"]  # shape: [batch_size, seq_len]
    print(f"Batch input_ids shape: {input_ids.shape}")

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)

    if len(activation_map) == 0:
        print("No LoRA sub-layer activations captured. Possibly no triggers for this input.")
        return

    # A function to fix shapes for A,B
    def fix_lora_by_input_shape(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray):
        """
        If x_data shape is e.g. (batch_size, seq_len, hidden_dim),
        hidden_dim = x_data.shape[-1]. We ensure A => [hidden_dim, r], B => [r, out_dim].
        """
        in_dim = x_data.shape[-1]
        a0, a1 = A.shape
        # A => [in_dim, r]
        if a0 == in_dim:
            r = a1
        elif a1 == in_dim:
            A = A.transpose(0,1)
            r = A.shape[1]
        else:
            raise ValueError(f"A shape {A.shape} doesn't match x_data last dim {in_dim}.")

        b0, b1 = B.shape
        if b0 == r:
            out_dim = b1
        elif b1 == r:
            B = B.transpose(0,1)
            out_dim = B.shape[1]
        else:
            raise ValueError(f"B shape {B.shape} doesn't match rank={r} in any dimension.")

        return A, B, in_dim, r, out_dim

    # Flattened submodule that expects shape [batch_size, seq_len * hidden_dim]
    class LoraApplyModelFlattened(nn.Module):
        def __init__(self, A, B, batch_size: int, seq_len: int, hidden_dim: int):
            super().__init__()
            self.register_buffer("A", A)
            self.register_buffer("B", B)
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim

        def forward(self, x_2d):
            # x_2d => shape [batch_size, seq_len * hidden_dim]
            # reshape => [batch_size, seq_len, hidden_dim]
            x_3d = x_2d.view(self.batch_size, self.seq_len, self.hidden_dim)
            out_3d = (x_3d @ self.A) @ self.B
            out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
            # For demonstration, flatten the output to [batch_size, -1]
            out_2d = out_3d.view(self.batch_size, -1)
            return out_2d

    # If user only wants certain submodules (like "attn.c_attn"), filter
    submodule_items = list(activation_map.items())
    if submodule_key:
        submodule_items = [(k,v) for (k,v) in submodule_items if submodule_key in k]

    for full_name, x_data in submodule_items:
        # x_data shape e.g. [batch_size, seq_len, hidden_dim]
        # Flatten => [batch_size, seq_len*hidden_dim]
        if x_data.ndim != 3:
            print(f"Skipping {full_name} because hooking shape is not 3D: {x_data.shape}")
            continue

        batch_size = x_data.shape[0]
        seq_len    = x_data.shape[1]
        hidden_dim = x_data.shape[2]
        flat_2d = x_data.reshape(batch_size, seq_len * hidden_dim)

        # Extract A,B
        submodule = dict(model.named_modules()).get(full_name, None)
        if submodule is None:
            print(f"Could not find submodule named {full_name} in model dict, skipping.")
            continue

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
            print(f"LoRA A submodule for {full_name} has no .weight, skipping.")
            continue
        if not hasattr(B_mod, "weight"):
            print(f"LoRA B submodule for {full_name} has no .weight, skipping.")
            continue

        A_raw = A_mod.weight.detach().cpu().float()
        B_raw = B_mod.weight.detach().cpu().float()

        # fix shapes
        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_by_input_shape(A_raw, B_raw, x_data)
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            continue

        lora_mod = LoraApplyModelFlattened(A_fixed, B_fixed, batch_size, seq_len, hidden_dim).eval()

        # We'll produce ONNX + JSON
        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        x_tensor = torch.from_numpy(flat_2d)
        # Export with fixed shape [batch_size, seq_len*hidden_dim]
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
                # no dynamic_axes => shape is fixed
                training=TrainingMode.TRAINING,
                keep_initializers_as_inputs=False
            )
        except Exception as e:
            print(f"Export error for {full_name}: {e}")
            continue

        # JSON
        data_json = {"input_data": flat_2d.tolist()}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path} with shape {flat_2d.shape}")
