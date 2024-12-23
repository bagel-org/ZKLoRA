import os
import json
import re
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
base_model_name = "distilgpt2"
lora_model_name = "ng0-k1/distilgpt2-finetuned-es"  # Adjust if different
output_dir = "lora_onnx_params"
json_dir = "intermediate_activations"
params_dir = "lora_params"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(params_dir, exist_ok=True)

# Input text for generating intermediate activations
input_text = "Hello, world!"
batch_size = 1
seq_len = 5   # Adjust if needed, or rely on actual input length
hidden_dim = 768  # DistilGPT2 hidden size

#############################################
# Load LoRA model
#############################################
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

#############################################
# Identify LoRA Layers
#############################################
lora_layers = []
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        print(name, module)
        lora_layers.append((name, module))

if not lora_layers:
    raise ValueError("No LoRA layers found in the model.")

# We'll parse layer index from the name if possible. For DistilGPT2:
# name might look like: base_model.model.transformer.h.0.attn.c_attn.lora_...
layer_idx_pattern = re.compile(r"transformer\.h\.(\d+)\.attn\.c_attn")

#############################################
# Capture intermediate activations x
# We'll register a hook on each transformer layer to capture its output
#############################################
intermediate_activations = {}

def make_hook(idx):
    def hook(module, inp, out):
        # If out is a tuple, the hidden states are usually at out[0].
        # Check the type of out and extract accordingly:
        if isinstance(out, tuple):
            hidden_states = out[0]
        else:
            hidden_states = out
        
        intermediate_activations[idx] = hidden_states.detach().cpu().numpy()
    return hook


# Register hooks for each layer
transformer_layers = model.base_model.model.transformer.h
for i, layer in enumerate(transformer_layers):
    layer.register_forward_hook(make_hook(i))

# Run forward to populate intermediate_activations
with torch.no_grad():
    _ = model(input_ids)

# Check we got activations for all layers
if len(intermediate_activations) == 0:
    raise ValueError("No intermediate activations captured. Check hook placement.")

#############################################
# For each LoRA layer:
# - Extract A, B
# - Save A and B as npy
# - Extract x for that layer (from intermediate_activations)
# - Export ONNX with (x, A, B) as inputs
# - Create JSON input file
# - Print ONNX inputs and do a basic validation
#############################################
def extract_lora_weights(lora_module):
    A = lora_module.lora_A['default'].weight.detach().cpu().float()
    B = lora_module.lora_B['default'].weight.detach().cpu().float()
    return A, B



class LoraApplyModel(nn.Module):
    def forward(self, x, A, B):
        # Ensure A is [768,4], B is [4,2304]
        if A.shape == (4, 768):
            A = A.transpose(0, 1) # [768,4]
        if B.shape == (2304, 4):
            B = B.transpose(0, 1) # [4,2304]

        # x: [batch, seq_len, 768]
        # (x @ A): [batch, seq_len, 4]
        # ((x @ A) @ B): [batch, seq_len, 2304]
        out = (x @ A) @ B

        # Add a dependency on x, A, B to prevent folding (optional)
        #out = out + x.mean() + A.sum() + B.sum()
        return out


def load_onnx_input_specs(onnx_path):
    m = onnx.load(onnx_path)
    graph = m.graph
    onnx_type_to_numpy = {
        1: np.float32, 2: np.uint8, 3: np.int8, 4: np.uint16, 5: np.int16,
        6: np.int32, 7: np.int64, 9: np.bool_, 10: np.float16, 11: np.double,
        12: np.uint32, 13: np.uint64, 14: np.complex64, 15: np.complex128,
        16: np.float64,
    }
    inputs = []
    for inp in graph.input:
        ttype = inp.type.tensor_type
        shape = [d.dim_value for d in ttype.shape.dim]
        dtype = onnx_type_to_numpy[ttype.elem_type]
        inputs.append((inp.name, shape, dtype))
    return inputs

all_valid = True

for (name, lora_module) in lora_layers:
    match = layer_idx_pattern.search(name)
    if not match:
        # If we can't find a layer index in the name, we can assign a dummy index or skip
        # For simplicity, skip layers that don't match the pattern
        print(f"Could not extract layer index from {name}, skipping.")
        continue
    layer_idx = int(match.group(1))

    # Extract A, B
    A, B = extract_lora_weights(lora_module)
    # Save A, B as npy
    A_path = os.path.join(params_dir, f"A_layer{layer_idx}.npy")
    B_path = os.path.join(params_dir, f"B_layer{layer_idx}.npy")
    np.save(A_path, A.numpy())
    np.save(B_path, B.numpy())

    # Get x from intermediate activations
    if layer_idx not in intermediate_activations:
        print(f"No intermediate activations for layer {layer_idx}")
        all_valid = False
        continue
    x_data = intermediate_activations[layer_idx]

    # Create a model and export ONNX
    lora_mod = LoraApplyModel().eval()
    x_tensor = torch.from_numpy(x_data)
    A_tensor = A.clone()
    B_tensor = B.clone()

    onnx_path = os.path.join(output_dir, f"layer_{layer_idx}_lora.onnx")

    torch.onnx.export(
        lora_mod,
        (x_tensor, A_tensor, B_tensor),
        onnx_path,
        export_params=False,
        do_constant_folding=False,
        opset_version=11,
        input_names=["input_x", "input_A", "input_B"],
        output_names=["output"],
        dynamic_axes={
            "input_x": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"}
        },
        training=TrainingMode.TRAINING
    )

    # Check inputs
    onnx_model = onnx.load(onnx_path)
    print(f"For {onnx_path}, ONNX Inputs:", onnx_model.graph.input)

    # Create JSON with [x, A, B]
    data_json = {
        "input_data": [x_data.tolist(), A.numpy().tolist(), B.numpy().tolist()]
    }
    json_path = os.path.join(json_dir, f"layer_{layer_idx}_key_projection.json")
    with open(json_path, "w") as f:
        json.dump(data_json, f)
    print(f"Saved JSON at {json_path}")

    # Basic validation
    input_specs = load_onnx_input_specs(onnx_path)
    if not input_specs:
        print(f"No inputs found in ONNX model {onnx_path}")
        all_valid = False
    else:
        # Check number of inputs
        if len(input_specs) != 3:
            print(f"Expected 3 inputs (x, A, B) but got {len(input_specs)} in {onnx_path}")
            all_valid = False
        else:
            # Optional: Check shapes if they are fixed
            # For now, just print a success message
            print(f"Validation: ONNX model {onnx_path} has {len(input_specs)} inputs as expected.")

if all_valid:
    print("All layers processed and validated successfully.")
else:
    print("Some layers failed validation.")
