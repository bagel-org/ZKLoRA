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
lora_model_name = "ng0-k1/distilgpt2-finetuned-es"
output_dir = "lora_onnx_params_params_inside"
json_dir = "intermediate_activations_params_inside"
params_dir = "lora_params_params_inside"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(json_dir, exist_ok=True)
os.makedirs(params_dir, exist_ok=True)

input_text = "Hello, world!"
batch_size = 1
seq_len = 5
hidden_dim = 768  # DistilGPT2 hidden dim

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

layer_idx_pattern = re.compile(r"transformer\.h\.(\d+)\.attn\.c_attn")

#############################################
# Capture intermediate activations
#############################################
intermediate_activations = {}

def make_hook(idx):
    def hook(module, inp, out):
        # out might be tuple
        if isinstance(out, tuple):
            hidden_states = out[0]
        else:
            hidden_states = out
        intermediate_activations[idx] = hidden_states.detach().cpu().numpy()
    return hook

transformer_layers = model.base_model.model.transformer.h
for i, layer in enumerate(transformer_layers):
    layer.register_forward_hook(make_hook(i))

with torch.no_grad():
    _ = model(input_ids)

if len(intermediate_activations) == 0:
    raise ValueError("No intermediate activations captured.")

#############################################
# Extract LoRA weights
#############################################
def extract_lora_weights(lora_module):
    A = lora_module.lora_A['default'].weight.detach().cpu().float()
    B = lora_module.lora_B['default'].weight.detach().cpu().float()
    return A, B

#############################################
# Model with A, B as parameters
#############################################
class LoraApplyModelParams(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        # Transpose them to correct shape
        if A.shape == (4, 768):
            A = A.transpose(0,1)  # [768,4]
        if B.shape == (2304,4):
            B = B.transpose(0,1)  # [4,2304]

        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, x):
        out = (x @ self.A) @ self.B
        out = out + x.mean() + self.A.sum() + self.B.sum()
        return out

#############################################
# ONNX Input Specs Helper
#############################################
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

#############################################
# Process each LoRA layer
#############################################
all_valid = True

for (name, lora_module) in lora_layers:
    match = layer_idx_pattern.search(name)
    if not match:
        print(f"Could not extract layer index from {name}, skipping.")
        continue
    layer_idx = int(match.group(1))

    A, B = extract_lora_weights(lora_module)
    # Save A,B as npy just for reference
    A_path = os.path.join(params_dir, f"A_layer{layer_idx}.npy")
    B_path = os.path.join(params_dir, f"B_layer{layer_idx}.npy")
    np.save(A_path, A.numpy())
    np.save(B_path, B.numpy())

    if layer_idx not in intermediate_activations:
        print(f"No intermediate activations for layer {layer_idx}")
        all_valid = False
        continue
    x_data = intermediate_activations[layer_idx]

    # Create model with A,B as parameters
    lora_mod = LoraApplyModelParams(A.clone(), B.clone()).eval()
    x_tensor = torch.from_numpy(x_data)

    onnx_path = os.path.join(output_dir, f"layer_{layer_idx}_lora_params_inside.onnx")

    # Here we embed A, B as parameters inside the ONNX file
    # by using export_params=True and keep_initializers_as_inputs=False.
    torch.onnx.export(
        lora_mod,
        x_tensor,
        onnx_path,
        export_params=True,  # embed parameters
        do_constant_folding=False,
        opset_version=11,
        input_names=["input_x"],
        output_names=["output"],
        dynamic_axes={
            "input_x": {0: "batch_size", 1: "seq_len"},
            "output": {0: "batch_size", 1: "seq_len"}
        },
        training=TrainingMode.TRAINING,
        keep_initializers_as_inputs=False  # ensure parameters are not considered inputs
    )

    # Check inputs
    onnx_model = onnx.load(onnx_path)
    print(f"For {onnx_path}, ONNX Inputs:", onnx_model.graph.input)

    # Create JSON with [x]
    data_json = {
        "input_data": [x_data.tolist()]
    }
    json_path = os.path.join(json_dir, f"layer_{layer_idx}_key_projection.json")
    with open(json_path, "w") as f:
        json.dump(data_json, f)
    print(f"Saved JSON at {json_path}")

    input_specs = load_onnx_input_specs(onnx_path)
    if not input_specs:
        print(f"No inputs found in ONNX model {onnx_path}, which is expected if A,B are constants.")
        # If no inputs means x also got optimized away, that would be odd. 
        # Check if you want at least x as input. If x is input, we should see it.
        all_valid = False
    else:
        # We expect only one input: x
        if len(input_specs) == 1 and input_specs[0][0] == "input_x":
            print(f"Validation: ONNX model {onnx_path} has only 'x' as input. A,B are embedded as constants.")
        else:
            print(f"Unexpected inputs: {input_specs}. Expected only 'input_x'.")
            all_valid = False

if all_valid:
    print("All layers processed and validated successfully.")
else:
    print("Some layers failed validation.")
