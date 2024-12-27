#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture LoRA submodule inputs, flatten them, export a flattened ONNX sub-module, and write JSON for EZKL."
    )
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path (e.g. 'distilgpt2').")
    parser.add_argument("--lora_model", type=str, required=True,
                        help="LoRA model name/path (e.g. 'some/lora-adapter').")
    parser.add_argument("--input_text", type=str, default="Hello, world!",
                        help="Sample text for capturing sub-layer activations.")
    parser.add_argument("--output_dir", type=str, default="lora_onnx_params",
                        help="Directory to save ONNX files.")
    parser.add_argument("--json_dir", type=str, default="intermediate_activations",
                        help="Directory to save JSON input files.")
    parser.add_argument("--submodule_key", type=str, default=None,
                        help="If provided, only export submodules whose name contains this key.")
    return parser.parse_args()

def main():
    args = parse_args()

    base_model_name = args.base_model
    lora_model_name = args.lora_model
    input_text = args.input_text
    output_dir = args.output_dir
    json_dir = args.json_dir
    submodule_key = args.submodule_key

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    print(f"Loading LoRA model: {lora_model_name}")
    model = PeftModel.from_pretrained(base_model, lora_model_name)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Example input
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs["input_ids"]
    print(f"Input IDs shape: {input_ids.shape}")  # e.g. (1, 4) if you have 4 tokens

    # We'll store sub-layer inputs in a dictionary
    activation_map = {}
    issued_wte_warning = False

    def register_lora_hooks_recursive(model, activation_map):
        """
        Recursively finds LoRA submodules. If submodule is 'wte'/'wpe', skip hooking.
        Otherwise, register a forward hook capturing the sub-layer input (x).
        """
        for full_name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if "wte" in full_name or "wpe" in full_name:
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
                        # Store the numpy array in activation_map
                        activation_map[mod_name] = x.detach().cpu().numpy()
                    return hook
                module.register_forward_hook(make_hook(full_name))

    register_lora_hooks_recursive(model, activation_map)

    # Run a forward pass
    with torch.no_grad():
        _ = model(input_ids)

    if len(activation_map) == 0:
        print("No LoRA sub-layer activations captured. Possibly no triggers for this input text.")
        return

    # We'll define a function to fix shapes of A, B
    def fix_lora_by_input_shape(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray):
        """
        E.g., x_data might be shape (1, 4, 768).
        We'll parse x_data.shape[-1] => 768, ensuring A => [768, r], B => [r, out_dim].
        """
        in_dim = x_data.shape[-1]
        a0, a1 = A.shape
        # Make A => [in_dim, r]
        if a0 == in_dim:
            r = a1
        elif a1 == in_dim:
            A = A.transpose(0,1)
            r = A.shape[1]
        else:
            raise ValueError(f"A shape {A.shape} doesn't match x_data last dim {in_dim} in any dimension.")

        b0, b1 = B.shape
        if b0 == r:
            out_dim = b1
        elif b1 == r:
            B = B.transpose(0,1)
            out_dim = B.shape[1]
        else:
            raise ValueError(f"B shape {B.shape} doesn't match rank={r} in any dimension.")

        return A, B, in_dim, r, out_dim

    # This submodule expects a 2D input [1, flatten_size]. We'll reshape internally to [1, seq_len, hidden_dim].
    class LoraApplyModelFlattened(nn.Module):
        def __init__(self, A, B, seq_len: int, hidden_dim: int):
            super().__init__()
            self.register_buffer("A", A)
            self.register_buffer("B", B)
            self.seq_len = seq_len
            self.hidden_dim = hidden_dim

        def forward(self, x_2d):
            # x_2d shape: [1, seq_len*hidden_dim]
            # reshape => [1, seq_len, hidden_dim]
            x_3d = x_2d.view(1, self.seq_len, self.hidden_dim)
            out_3d = (x_3d @ self.A) @ self.B
            out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
            # flatten output or not; let's flatten for demonstration
            out_2d = out_3d.view(1, -1)
            return out_2d

    # We'll do one loop: either we export all submodules or filter by submodule_key
    # Each submodule hooking data is shape (1, seq_len, hidden_dim) or (1, seq_len, 3072).
    all_valid = True

    for full_name, x_data in activation_map.items():
        if submodule_key and submodule_key not in full_name:
            continue

        # Grab the submodule from the model
        submodule = dict(model.named_modules()).get(full_name, None)
        if submodule is None:
            print(f"No submodule found for {full_name}, skipping.")
            continue

        # We assume x_data is e.g. shape [1,4,768], so flatten to [1,4*768=3072].
        flatten_shape = (1, x_data.shape[1]*x_data.shape[2])
        x_flat = x_data.reshape(flatten_shape)

        # Extract LoRA parameters
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

        # Fix shapes
        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_by_input_shape(A_raw, B_raw, x_data)
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            all_valid = False
            continue

        # Build the flattened model: e.g. shape x_2d => [1, 4*768], then .view(1,4,768)
        seq_len = x_data.shape[1]
        hidden_dim = x_data.shape[2]
        lora_mod = LoraApplyModelFlattened(A_fixed, B_fixed, seq_len, hidden_dim).eval()

        # We'll now create a dummy input for ONNX of shape [1, 4*768]
        x_tensor = torch.from_numpy(x_flat)

        # Export
        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        try:
            # No dynamic_axes => fixed shape (1, seq_len*hidden_dim)
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
            all_valid = False
            continue

        # Write the flattened JSON. Now we have shape e.g. (1,3072).
        # EZKL is more likely to parse this 2D shape than a 3D one.
        data_json = {"input_data": x_flat.tolist()}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path} (with shape {x_flat.shape})")

    if all_valid:
        print("All requested submodules processed successfully.")
    else:
        print("Some submodules failed validation.")


if __name__ == "__main__":
    main()
