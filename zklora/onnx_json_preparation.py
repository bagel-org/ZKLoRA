import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode
from peft import PeftModel
from transformers import PreTrainedModel, PreTrainedTokenizer


def export_lora_submodules_flattened(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    output_dir: str = "lora_onnx_params",
    json_dir: str = "intermediate_activations",
    submodule_key: str = None
) -> None:
    """
    Captures LoRA sub-layer activations from a PeftModel, flattens them, and exports each
    submodule to an ONNX file with shape [1, seq_len*hidden_dim], along with a flattened JSON
    that can be parsed by EZKL.

    :param model:          A PEFT (LoRA) model, already loaded and in eval mode.
    :param tokenizer:      The corresponding tokenizer for generating input IDs.
    :param input_text:     Sample text for capturing sub-layer activations.
    :param output_dir:     Directory where ONNX files will be saved.
    :param json_dir:       Directory where JSON input files will be saved.
    :param submodule_key:  If set, only export submodules whose name contains this string.
    """

    # Make sure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    # We'll store sub-layer inputs (activations) in a dictionary
    activation_map = {}
    issued_wte_warning = False

    def register_lora_hooks_recursive(model: PreTrainedModel, activation_map: dict):
        """
        Recursively finds LoRA submodules. If submodule name has 'wte'/'wpe', skip hooking.
        Otherwise, register a forward hook that captures the sub-layer input (x).
        """
        for full_name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Skip if submodule name has wte/wpe
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
                        x = layer_inputs[0]
                        print(f"shape in hook ({mod_name}):", x.size())
                        activation_map[mod_name] = x.detach().cpu().numpy()
                    return hook

                module.register_forward_hook(make_hook(full_name))

    # Hook all LoRA submodules in the model
    register_lora_hooks_recursive(model, activation_map)

    # Now produce the input_ids from the sample text
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs["input_ids"]

    # Forward pass to populate activation_map
    with torch.no_grad():
        _ = model(input_ids)

    if len(activation_map) == 0:
        print("No LoRA sub-layer activations captured. Possibly no triggers for this input.")
        return

    # A helper to fix shapes for A, B
    def fix_lora_by_input_shape(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray):
        """
        Ensures A => [in_dim, r], B => [r, out_dim], derived from x_data shape. 
        If x_data is (1, seq_len, hidden_dim), then hidden_dim= x_data.shape[-1].
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

    class LoraApplyModelFlattened(nn.Module):
        """
        Expects flattened input of shape (1, seq_len*hidden_dim).
        Internally reshapes to (1, seq_len, hidden_dim), does (x @ A) @ B, etc.
        """
        def __init__(self, A, B, seq_len: int, hidden_dim: int):
            super().__init__()
            self.register_buffer("A", A)
            self.register_buffer("B", B)
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim

        def forward(self, x_2d):
            # x_2d: shape [1, seq_len * hidden_dim]
            x_3d = x_2d.view(1, self.seq_len, self.hidden_dim)
            out_3d = (x_3d @ self.A) @ self.B
            out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
            # Optionally flatten the output or keep it 3D. We'll flatten for demonstration.
            out_2d = out_3d.view(1, -1)
            return out_2d

    # Now loop over submodules we captured
    for full_name, x_data in activation_map.items():
        if submodule_key and submodule_key not in full_name:
            continue

        # Gather the submodule from the model
        submodule = dict(model.named_modules()).get(full_name, None)
        if submodule is None:
            print(f"No submodule found for {full_name}, skipping.")
            continue

        # Flatten x_data from shape e.g. (1,4,768) => (1, 4*768)
        if x_data.ndim != 3:
            print(f"Skipping {full_name} because hooking shape is not 3D: {x_data.shape}")
            continue
        seq_len = x_data.shape[1]
        hidden_dim = x_data.shape[2]
        flattened_shape = (1, seq_len * hidden_dim)
        x_flat = x_data.reshape(flattened_shape)

        # Extract A,B from lora_A, lora_B
        if hasattr(submodule.lora_A, "keys"):
            a_keys = list(submodule.lora_A.keys())
            if not a_keys:
                print(f"No keys in submodule.lora_A for {full_name}, skipping.")
                continue
            a_key = a_keys[0]
            A_mod = submodule.lora_A[a_key]
        else:
            A_mod = submodule.lora_A

        if hasattr(submodule.lora_B, "keys"):
            b_keys = list(submodule.lora_B.keys())
            if not b_keys:
                print(f"No keys in submodule.lora_B for {full_name}, skipping.")
                continue
            b_key = b_keys[0]
            B_mod = submodule.lora_B[b_key]
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

        # Fix shapes
        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_by_input_shape(A_raw, B_raw, x_data)
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            continue

        # Build the flattened sub-module
        lora_mod = LoraApplyModelFlattened(A_fixed, B_fixed, seq_len, hidden_dim).eval()

        # Export ONNX
        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        x_tensor = torch.from_numpy(x_flat)
        try:
            # We'll fix the shape to [1, seq_len*hidden_dim], no dynamic axes
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
                keep_initializers_as_inputs=False
            )
        except Exception as e:
            print(f"Export error for {full_name}: {e}")
            continue

        # Write JSON with shape [1, seq_len*hidden_dim]
        data_json = {"input_data": x_flat.tolist()}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path} (shape {x_flat.shape})")


# --------------------------------------------------------------------
# Example usage from another script:
#
# from lora_export import export_lora_submodules_flattened
# from transformers import AutoModel, AutoTokenizer
# from peft import PeftModel
#
# base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
# lora_model = PeftModel.from_pretrained(base_model, "shirzady1934/distilgpt-monolinugal")
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# lora_model.eval()
#
# export_lora_submodules_flattened(
#     model=lora_model,
#     tokenizer=tokenizer,
#     input_text="Hello, world!",
#     output_dir="lora_onnx_params",
#     json_dir="intermediate_activations",
#     submodule_key=None
# )
#
# # Then you can run ezkl or do further processing with the exported ONNX + JSON files.
