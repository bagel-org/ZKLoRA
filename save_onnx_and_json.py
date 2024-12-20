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
# Step 1: Load PEFT (LoRA) model
#############################################
base_model_name = "distilgpt2"
lora_model_name = "ng0-k1/distilgpt2-finetuned-es"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Example input text
input_text = "Hello, world!"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

#############################################
# Step 2: Identify a LoRA layer and extract A,B
#############################################
lora_layers = []
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        lora_layers.append((name, module))

if not lora_layers:
    raise ValueError("No LoRA layers found in the model.")

# Just take the first LoRA layer for demonstration
lora_name, lora_module = lora_layers[0]

A_param = lora_module.lora_A['default'].weight.detach().cpu().float().clone()
B_param = lora_module.lora_B['default'].weight.detach().cpu().float().clone()

# Save A and B as npy
os.makedirs("lora_params", exist_ok=True)
np.save("lora_params/A.npy", A_param.numpy())
np.save("lora_params/B.npy", B_param.numpy())

#############################################
# Step 3: Run forward pass to get intermediate activations x
#############################################
# We need "intermediate activations" from a certain point in the model.
# For simplicity, let's hook into the output of the first transformer layer.
intermediate_activations = {}

def capture_hook(module, input, output):
    # output is the hidden states after this layer
    intermediate_activations['x'] = output.detach().cpu().numpy()

# Hook into the first layer output
model.base_model.model.transformer.h[0].register_forward_hook(capture_hook)

with torch.no_grad():
    _ = model(input_ids)

if 'x' not in intermediate_activations:
    raise ValueError("No intermediate activations captured.")

x_data = intermediate_activations['x']  # shape: (batch, seq_len, hidden_dim)

# Assume DistilGPT2 hidden_dim=768, batch_size=1, seq_len=?
batch_size, seq_len, hidden_dim = x_data.shape

#############################################
# Step 4: We already saved A and B as npy files.
# Step 5: Define a new model that takes x, A, B as inputs
#############################################
class LoraApplyModel(nn.Module):
    def forward(self, x, A, B):
        # Adjust A and B shapes if needed
        in_dim = x.shape[-1]
        if B.shape[0] != in_dim:
            B = B.transpose(0, 1)
        if A.shape[0] != B.shape[1]:
            A = A.transpose(0, 1)

        out = torch.matmul(torch.matmul(x, B), A)
        # Depend on x, A, B in a non-trivial way
        out = out + x.mean() + A.sum() + B.sum()
        return out

model_for_onnx = LoraApplyModel().eval()

# Load A and B back as tensors
A_loaded = torch.from_numpy(np.load("lora_params/A.npy"))
B_loaded = torch.from_numpy(np.load("lora_params/B.npy"))
x_tensor = torch.from_numpy(x_data)

#############################################
# Step 6: Export the new model to ONNX with x,A,B as inputs
#############################################
os.makedirs("lora_onnx_params", exist_ok=True)
onnx_path = "lora_onnx_params/lora_layer0.onnx"

torch.onnx.export(
    model_for_onnx,
    (x_tensor, A_loaded, B_loaded),
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

# Check if inputs are present
onnx_model = onnx.load(onnx_path)
print("ONNX Inputs:", onnx_model.graph.input)

#############################################
# Step 7: Create a JSON file with [x, A, B]
#############################################
# input_data: [x, A, B]
data_json = {
    "input_data": [x_data.tolist(), A_param.numpy().tolist(), B_param.numpy().tolist()]
}

json_path = "intermediate_activations/layer_0_key_projection.json"
os.makedirs("intermediate_activations", exist_ok=True)
with open(json_path, "w") as f:
    json.dump(data_json, f)

print(f"Saved JSON at {json_path}")

#############################################
# Step 8: Validate JSON against the ONNX model's input shape
#############################################
def load_onnx_input_specs(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
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

input_specs = load_onnx_input_specs(onnx_path)
if not input_specs:
    print("No inputs found in the ONNX model.")
else:
    with open(json_path, "r") as f:
        jd = json.load(f)
    input_data = jd["input_data"]

    if len(input_data) != len(input_specs):
        print(f"Input count mismatch: JSON has {len(input_data)} inputs, ONNX expects {len(input_specs)}.")
    else:
        all_good = True
        for (name, shape, dtype), arr in zip(input_specs, input_data):
            np_arr = np.array(arr, dtype=dtype)
            # Basic size check if shape is fixed
            if 0 not in shape:  # no dynamic axes
                expected_size = 1
                for dim in shape:
                    expected_size *= dim if dim > 0 else 1
                if np_arr.size != expected_size:
                    print(f"Size mismatch for {name}: expected {expected_size}, got {np_arr.size}")
                    all_good = False
                try:
                    np_arr.reshape(shape)
                except:
                    print(f"Could not reshape {name} to {shape}")
                    all_good = False
        if all_good:
            print("JSON input data successfully validated against ONNX model inputs.")
