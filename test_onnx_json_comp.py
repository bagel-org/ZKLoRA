import os
import json
import re
import numpy as np
import onnx

def load_onnx_input_shape(onnx_path):
    model = onnx.load(onnx_path)
    graph = model.graph
    if len(graph.input) == 0:
        raise ValueError(f"No inputs found in ONNX model: {onnx_path}")
    input_tensor = graph.input[0]

    # Extract shape and dtype
    input_type = input_tensor.type.tensor_type
    shape = [d.dim_value for d in input_type.shape.dim]
    dtype = input_type.elem_type

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

    np_dtype = onnx_type_to_numpy.get(dtype, None)
    if np_dtype is None:
        raise ValueError(f"Unsupported or unknown ONNX dtype {dtype} in {onnx_path}")

    return shape, np_dtype

def validate_json_against_onnx(json_path, onnx_path):
    expected_shape, expected_dtype = load_onnx_input_shape(onnx_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    if "input_data" not in data:
        raise ValueError(f"JSON file {json_path} does not contain 'input_data' key.")

    input_data = data["input_data"]
    input_array = np.array(input_data, dtype=expected_dtype)

    expected_size = 1
    for dim in expected_shape:
        if dim == 0:
            raise ValueError(f"Dynamic dimension in ONNX input shape {expected_shape}, cannot validate exact size.")
        expected_size *= dim

    input_array_flat = input_array.flatten()
    if input_array_flat.size != expected_size:
        raise ValueError(
            f"Data size mismatch in {json_path}. "
            f"Expected {expected_size} elements, got {input_array_flat.size}."
        )

    # Attempt to reshape
    try:
        input_array_reshaped = input_array_flat.reshape(expected_shape)
    except ValueError:
        raise ValueError(
            f"Could not reshape data from {json_path} of size {input_array_flat.size} to shape {expected_shape}."
        )

    print(f"Validation successful for {json_path} against {onnx_path}!")
    print(f"Shape: {expected_shape}, dtype: {expected_dtype}")

if __name__ == "__main__":
    json_dir = "intermediate_activations"
    onnx_dir = "lora_onnx_params"

    if not os.path.isdir(json_dir):
        print(f"Directory {json_dir} not found.")
        exit(1)
    if not os.path.isdir(onnx_dir):
        print(f"Directory {onnx_dir} not found.")
        exit(1)

    # Regex to extract the layer index i from the ONNX filename
    # ONNX files are named like: base_model_model_transformer_h_0_attn_c_attn.onnx
    pattern = re.compile(r"base_model_model_transformer_h_(\d+)_attn_c_attn\.onnx")

    onnx_files = [f for f in os.listdir(onnx_dir) if f.endswith(".onnx")]
    if not onnx_files:
        print("No ONNX files found in lora_onnx_params.")
        exit(1)

    all_valid = True
    for onnx_file in onnx_files:
        match = pattern.match(onnx_file)
        if not match:
            print(f"ONNX file {onnx_file} does not match the expected pattern.")
            all_valid = False
            continue

        i = match.group(1)  # Extract layer index as a string
        json_file = f"layer_{i}_key_projection.json"
        json_path = os.path.join(json_dir, json_file)
        onnx_path = os.path.join(onnx_dir, onnx_file)

        if not os.path.isfile(json_path):
            print(f"No corresponding JSON file for {onnx_path} found at {json_path}")
            all_valid = False
            continue

        try:
            validate_json_against_onnx(json_path, onnx_path)
        except ValueError as e:
            print(f"Validation failed for {json_path} against {onnx_path}: {e}")
            all_valid = False

    if all_valid:
        print("All JSON inputs successfully validated against their corresponding ONNX models.")
    else:
        print("Some inputs failed validation. Check the logs above for details.")
