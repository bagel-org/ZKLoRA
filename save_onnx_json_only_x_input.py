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
    parser = argparse.ArgumentParser(description="Export LoRA submodules to ONNX with fixed shape (batch, seq_len).")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model name/path (e.g. 'distilgpt2').")
    parser.add_argument("--lora_model", type=str, required=True,
                        help="LoRA model name/path (e.g. 'shirzady1934/distilgpt-monolinugal').")
    parser.add_argument("--input_text", type=str, default="Hello, world!",
                        help="Sample text for capturing sub-layer activations.")
    parser.add_argument("--output_dir", type=str, default="lora_onnx_params",
                        help="Directory for saving ONNX files.")
    parser.add_argument("--json_dir", type=str, default="intermediate_activations",
                        help="Directory for saving JSON input files.")
    parser.add_argument("--submodule_key", type=str, default=None,
                        help="If provided, only export submodules whose name contains this key.")
    return parser.parse_args()


def main():
    args = parse_args()

    base_model_name = args.base_model
    lora_model_name = args.lora_model
    output_dir = args.output_dir
    json_dir = args.json_dir
    input_text = args.input_text
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

    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs["input_ids"]

    # A dictionary to store input activations for each LoRA submodule
    activation_map = {}
    issued_wte_warning = False

    def register_lora_hooks_recursive(model, activation_map):
        """
        Finds LoRA submodules; skip hooking if submodule has 'wte'/'wpe'.
        Otherwise, forward-hook captures the sub-layer input (x).
        """
        for full_name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if ("wte" in full_name or "wpe" in full_name):
                    nonlocal issued_wte_warning
                    if not issued_wte_warning:
                        print(f"WARNING: Found LoRA submodule '{full_name}' (wte/wpe). Skipping hooking for embeddings.")
                        issued_wte_warning = True
                    continue

                print(f"Registering hook on LoRA submodule: {full_name}")
                def make_hook(mod_name):
                    def hook(mod, layer_inputs, layer_output):
                        if not layer_inputs:
                            return
                        x = layer_inputs[0]
                        print(f"shape in hook ({mod_name}):", x.size())
                        # Store the numpy version of x for that submodule
                        activation_map[mod_name] = x.detach().cpu().numpy()
                    return hook

                module.register_forward_hook(make_hook(full_name))

    register_lora_hooks_recursive(model, activation_map)

    # Run forward pass
    with torch.no_grad():
        _ = model(input_ids)

    if len(activation_map) == 0:
        print("No LoRA sub-layer activations captured. Possibly no triggers for this input text.")
        return

    def fix_lora_by_input_shape(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray):
        """
        Ensures A => [in_dim, r] and B => [r, out_dim], derived from x_data's last dim.
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
            raise ValueError(f"A shape {A.shape} doesn't match x_data last dim {in_dim} in any dimension.")

        # B => [r, out_dim]
        b0, b1 = B.shape
        if b0 == r:
            out_dim = b1
        elif b1 == r:
            B = B.transpose(0,1)
            out_dim = B.shape[1]
        else:
            raise ValueError(f"B shape {B.shape} doesn't match rank={r} in any dimension.")
        return A, B, in_dim, r, out_dim

    class LoraApplyModel(nn.Module):
        def __init__(self, A, B):
            super().__init__()
            self.register_buffer("A", A)
            self.register_buffer("B", B)
        def forward(self, x):
            out = (x @ self.A) @ self.B
            # small dependency to avoid constant folding
            out = out + x.mean() + self.A.sum() + self.B.sum()
            return out

    def load_onnx_input_specs(onnx_path: str):
        onnx_model = onnx.load(onnx_path)
        graph = onnx_model.graph
        # parse shapes
        onnx_type_to_numpy = {
            1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
            6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.double,
            12: np.uint32, 13: np.uint64, 14: np.complex64, 15: np.complex128,
            16: np.float64
        }
        inputs = []
        for inp in graph.input:
            ttype = inp.type.tensor_type
            shape = [d.dim_value for d in ttype.shape.dim]
            dtype = onnx_type_to_numpy[ttype.elem_type]
            inputs.append((inp.name, shape, dtype))
        return inputs

    all_valid = True

    # Loop over submodules that have captured x_data. Optionally filter by --submodule_key
    for full_name, x_data in activation_map.items():
        if submodule_key and submodule_key not in full_name:
            continue

        # Retrieve submodule
        submodule = dict(model.named_modules()).get(full_name, None)
        if submodule is None:
            print(f"No submodule named {full_name} found in model dict.")
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
            print(f"LoRA A submodule for {full_name} has no .weight; skipping.")
            continue
        if not hasattr(B_mod, "weight"):
            print(f"LoRA B submodule for {full_name} has no .weight; skipping.")
            continue

        A_raw = A_mod.weight.detach().cpu().float()
        B_raw = B_mod.weight.detach().cpu().float()

        # fix shapes
        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_by_input_shape(A_raw, B_raw, x_data)
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            all_valid = False
            continue

        # Build sub-model
        lora_mod = LoraApplyModel(A_fixed, B_fixed).eval()
        x_tensor = torch.from_numpy(x_data)

        # We'll do a fixed shape for the ONNX input
        shape_str = f"{x_data.shape}"  # e.g. (1,4,768)
        print(f"Exporting submodule {full_name} with input shape {shape_str}")

        # Export path
        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        try:
            # No dynamic_axes => fix the shape exactly to x_data.shape
            torch.onnx.export(
                lora_mod,
                x_tensor,
                onnx_path,
                export_params=True,
                do_constant_folding=False,
                opset_version=11,
                input_names=["input_x"],
                output_names=["output"],
                # No dynamic_axes here => shape is fixed
                training=TrainingMode.TRAINING,
                keep_initializers_as_inputs=False
            )
        except Exception as e:
            print(f"Export error for {full_name}: {e}")
            all_valid = False
            continue

        # Write JSON
        # x_data => shape [1, 4, 768], e.g.
        data_json = {"input_data": x_data.tolist()}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path}")

        # Validate shape
        input_specs = load_onnx_input_specs(onnx_path)
        if not input_specs:
            print(f"No inputs found in {onnx_path}. Expected 1 input_x.")
            all_valid = False
        else:
            if len(input_specs) == 1 and input_specs[0][0] == "input_x":
                print(f"ONNX model {onnx_path} has only 'input_x' as external input, shape = {input_specs[0][1]}")
            else:
                print(f"Unexpected inputs in {onnx_path}: {input_specs}")
                all_valid = False

    if all_valid:
        print("All requested submodules processed successfully.")
    else:
        print("Some submodules failed validation.")


if __name__ == "__main__":
    main()
