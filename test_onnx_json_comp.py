import os
import json
import numpy as np
import onnx
import re

json_dir = "intermediate_activations"
onnx_dir = "lora_onnx_params"

# Regex to extract layer index from ONNX filename
# ONNX files look like: base_model_model_transformer_h_0_attn_c_attn.onnx
pattern = re.compile(r"base_model_model_transformer_h_(\d+)_attn_c_attn\.onnx")

onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
if not onnx_files:
    print("No ONNX files found.")
    exit(1)

all_valid = True

def load_onnx_input_specs(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    inputs = graph.input
    # Return a list of (name, shape, dtype)
    # dtype is ONNX elem_type converted to np dtype
    onnx_type_to_numpy = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        4: np.uint16,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        9: np.bool_,
        10: np.float16,
        11: np.double,
        12: np.uint32,
        13: np.uint64,
        14: np.complex64,
        15: np.complex128,
        16: np.float64,
    }
    input_specs = []
    for inp in inputs:
        ttype = inp.type.tensor_type
        shape = []
        for d in ttype.shape.dim:
            # dim_value = 0 often means dynamic, handle accordingly
            shape.append(d.dim_value)
        dtype = onnx_type_to_numpy[ttype.elem_type]
        input_specs.append((inp.name, shape, dtype))
    return input_specs

for onnx_file in onnx_files:
    match = pattern.match(onnx_file)
    if not match:
        print(f"ONNX file {onnx_file} does not match the expected pattern.")
        all_valid = False
        continue

    i = match.group(1)
    json_file = f"layer_{i}_key_projection.json"
    json_path = os.path.join(json_dir, json_file)
    onnx_path = os.path.join(onnx_dir, onnx_file)

    if not os.path.isfile(json_path):
        print(f"No corresponding JSON file for {onnx_path} found at {json_path}")
        all_valid = False
        continue

    # Load ONNX input specs
    try:
        input_specs = load_onnx_input_specs(onnx_path)
    except Exception as e:
        print(f"Error loading ONNX model {onnx_path}: {e}")
        all_valid = False
        continue

    if not input_specs:
        print(f"Validation failed for {json_path} against {onnx_path}: No inputs found in ONNX model.")
        all_valid = False
        continue

    # Load JSON
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "input_data" not in data:
        print(f"JSON file {json_path} does not contain 'input_data'.")
        all_valid = False
        continue

    input_data = data["input_data"]
    # input_data should be a list of arrays: [x, A, B]
    if len(input_data) != len(input_specs):
        print(f"Number of inputs in {json_path} ({len(input_data)}) does not match ONNX inputs ({len(input_specs)}).")
        all_valid = False
        continue

    # Validate each input
    for (name, shape, dtype), arr in zip(input_specs, input_data):
        np_arr = np.array(arr, dtype=dtype)
        # Check total size
        expected_size = 1
        for d in shape:
            if d == 0:
                # Dynamic dimension, we won't strictly check size here.
                # Just ensure that we can at least reshape by inferring that dimension.
                # For simplicity, ignore strict checks for dynamic dim.
                continue
            expected_size *= d if d > 0 else 1

        # If there are no dynamic dims, we can check size strictly
        if 0 not in shape:
            flat = np_arr.flatten()
            if flat.size != expected_size:
                print(f"Size mismatch for input {name} in {json_path}. Expected {expected_size}, got {flat.size}")
                all_valid = False
                continue
            # Try reshape
            try:
                flat.reshape(shape)
            except Exception:
                print(f"Could not reshape input {name} in {json_path} to {shape}")
                all_valid = False
                continue

    if all_valid:
        print(f"Validation successful for {json_path} against {onnx_path}!")

if all_valid:
    print("All JSON inputs successfully validated against their corresponding ONNX models.")
else:
    print("Some inputs failed validation. Check the logs above for details.")
