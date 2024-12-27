import json
import struct
import numpy as np

def flatten_list(nested_list):
    if not isinstance(nested_list, list):
        return [nested_list]
    if len(nested_list) == 0:
        return []
    if isinstance(nested_list[0], list):
        return flatten_list(nested_list[0])
    return nested_list

def float32_to_uint64(activations):
    # Convert float32 activations to uint64 encoding
    float32_array = np.array(activations, dtype=np.float32)
    uint64_encoded = []
    
    for float_val in float32_array:
        # Pack float32 into bytes then interpret as uint64
        bytes_val = struct.pack('f', float_val)
        uint64_val = struct.unpack('Q', bytes_val + b'\x00'*4)[0]  # Pad to 8 bytes
        uint64_encoded.append(uint64_val)
        
    return uint64_encoded

if __name__ == "__main__":
    
    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    with open(activations_path, "r") as f:
        activations = json.load(f)

    activations = flatten_list(activations["input_data"])

    uint64_activations = float32_to_uint64(activations)

    
    
    