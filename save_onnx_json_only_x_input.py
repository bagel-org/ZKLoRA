import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#############################################
# Configuration
#############################################
base_model_name = "distilgpt2"  # or something else
lora_model_name = "shirzady1934/distilgpt-monolinugal"  
output_dir = "lora_onnx_params"
json_dir = "intermediate_activations"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)

input_text = "Hello, world!"

#############################################
# 1. Load Base + LoRA Model
#############################################
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(input_text, return_tensors='pt')
input_ids = inputs["input_ids"]

#############################################
# 2. Activation map to store sub-layer inputs
#############################################
activation_map = {}

issued_wte_warning = False  # track if we've warned about wte/wpe

def register_lora_hooks_recursive(model, activation_map) -> None:
    """
    Recursively finds LoRA submodules, checks if submodule name has "wte"/"wpe".
    If so, skip hooking and optionally print a warning once.
    Otherwise, register a forward hook that captures the input for that submodule.
    """
    for full_name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # If submodule is named wte or wpe => skip hooking
            if ("wte" in full_name or "wpe" in full_name):
                global issued_wte_warning
                if not issued_wte_warning:
                    print(
                        f"WARNING: Found LoRA submodule '{full_name}' (wte/wpe). "
                        "Our code doesn't support embedding LoRA. Skipping hooking."
                    )
                    issued_wte_warning = True
                continue

            # Otherwise, register the hook
            print(f"Registering hook on LoRA submodule: {full_name}")
            def make_hook(mod_name):
                def hook(mod, layer_inputs, layer_output):
                    if not layer_inputs:
                        return
                    x = layer_inputs[0]
                    activation_map[mod_name] = x.detach().cpu().numpy()
                return hook
            module.register_forward_hook(make_hook(full_name))

# Register hooks (and skip wte/wpe entirely)
register_lora_hooks_recursive(model, activation_map)

#############################################
# 3. Forward pass
#############################################
with torch.no_grad():
    _ = model(input_ids)

if len(activation_map) == 0:
    print("No sub-layer activations captured. Possibly no LoRA submodules were triggered by this input.")

#############################################
# The rest of your script remains unchanged:
# fix_lora_by_input_shape, LoraApplyModel, load_onnx_input_specs, 
# main loop extracting A,B, building sub-model, etc.
#############################################

def fix_lora_by_input_shape(A: torch.Tensor, B: torch.Tensor, x_data: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, int, int, int]:
    in_dim = x_data.shape[-1]
    a0, a1 = A.shape
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

class LoraApplyModel(nn.Module):
    def __init__(self, A, B) -> None:
        super().__init__()
        self.register_buffer("A", A)
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x @ self.A) @ self.B
        out = out + x.mean() + self.A.sum() + self.B.sum()
        return out

def load_onnx_input_specs(onnx_path: str) -> list[tuple[str, list[int], np.dtype]]:
    m = onnx.load(onnx_path)
    graph = m.graph
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

# Main loop: submodules with wte/wpe were never hooked, so they won't appear in activation_map.
for full_name, submodule in model.named_modules():
    if hasattr(submodule, 'lora_A') and hasattr(submodule, 'lora_B'):
        if ("wte" in full_name or "wpe" in full_name):
            # We already skipped hooking it, so skip it here too
            continue

        # Attempt to find a key in lora_A, lora_B
        if hasattr(submodule.lora_A, 'keys'):
            a_keys = list(submodule.lora_A.keys())
            if not a_keys:
                print(f"No keys found in submodule.lora_A for {full_name}, skipping.")
                continue
            a_key = a_keys[0]
            A_mod = submodule.lora_A[a_key]
        else:
            A_mod = submodule.lora_A

        if hasattr(submodule.lora_B, 'keys'):
            b_keys = list(submodule.lora_B.keys())
            if not b_keys:
                print(f"No keys found in submodule.lora_B for {full_name}, skipping.")
                continue
            b_key = b_keys[0]
            B_mod = submodule.lora_B[b_key]
        else:
            B_mod = submodule.lora_B

        if hasattr(A_mod, 'weight'):
            A_raw = A_mod.weight.detach().cpu().float()
        else:
            print(f"LoRA A mod for {full_name} has no .weight, skipping.")
            continue
        if hasattr(B_mod, 'weight'):
            B_raw = B_mod.weight.detach().cpu().float()
        else:
            print(f"LoRA B mod for {full_name} has no .weight, skipping.")
            continue

        x_data = activation_map.get(full_name, None)
        if x_data is None:
            print(f"No activation data for {full_name}. Possibly not triggered by this input.")
            all_valid = False
            continue

        try:
            A_fixed, B_fixed, in_dim, rank, out_dim = fix_lora_by_input_shape(A_raw, B_raw, x_data)
        except ValueError as e:
            print(f"Shape fix error for {full_name}: {e}")
            all_valid = False
            continue

        lora_mod = LoraApplyModel(A_fixed, B_fixed).eval()
        x_tensor = torch.from_numpy(x_data)

        safe_name = full_name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

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
                dynamic_axes={"input_x": {0: "batch_size"}, "output": {0: "batch_size"}},
                training=TrainingMode.TRAINING,
                keep_initializers_as_inputs=False
            )
        except Exception as e:
            print(f"Export error for {full_name}: {e}")
            all_valid = False
            continue

        data_json = {"input_data": [x_data.tolist()]}
        json_path = os.path.join(json_dir, f"{safe_name}.json")
        with open(json_path, "w") as f:
            json.dump(data_json, f)

        print(f"Exported ONNX for {full_name} -> {onnx_path}")
        print(f"Saved JSON -> {json_path}")

        input_specs = load_onnx_input_specs(onnx_path)
        if not input_specs:
            print(f"No inputs found in {onnx_path}. Expected 1 input_x.")
            all_valid = False
        else:
            if len(input_specs) == 1 and input_specs[0][0] == "input_x":
                print(f"ONNX model {onnx_path} has only 'input_x' as external input. Good!")
            else:
                print(f"Unexpected inputs in {onnx_path}: {input_specs}")
                all_valid = False

if all_valid:
    print("All submodules processed successfully.")
else:
    print("Some submodules failed validation.")
