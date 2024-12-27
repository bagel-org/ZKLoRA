import onnx
import ezkl
import time
import numpy as np
import asyncio
import json
import onnxruntime

def flatten_list(nested_list):
    if not isinstance(nested_list, list):
        return [nested_list]
    if len(nested_list) == 0:
        return []
    if isinstance(nested_list[0], list):
        return flatten_list(nested_list[0])
    return nested_list

async def main():
    lora_path = "lora_onnx_params/base_model_model_transformer_h_2_attn_c_attn.onnx"
    lora_ezkl = "model.ezkl"
    activations_path = "intermediate_activations/base_model_model_transformer_h_2_attn_c_attn.json"
    
    onnx_model = onnx.load(lora_path)
    param_count = sum(np.prod(param.dims) for param in onnx_model.graph.initializer)
    print(f"Number of parameters in {lora_path}: {param_count:,}")

    #
    # Setup
    #
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"
    py_run_args.logrows = 20

    print("Generating settings...")
    start_time = time.time()
    ezkl.gen_settings(lora_path, py_run_args=py_run_args)
    ezkl.compile_circuit(lora_path, lora_ezkl, "settings.json")
    ezkl.gen_srs("kzg.srs", py_run_args.logrows)
    ezkl.setup(lora_ezkl, "vk.key", "pk.key", "kzg.srs")
    end_time = time.time()
    print(f"Setup took {end_time - start_time:.2f} seconds")

    #
    # Prove
    #
    # Load the sub-layer activations
    with open(activations_path, "r") as f:
        activations_data = json.load(f)
    
    print("Raw JSON keys:", activations_data.keys())

    # Convert to float32 array
    float_data = np.array(activations_data["input_data"], dtype=np.float32)
    print("Activations shape (from JSON):", float_data.shape)

    # 1) Check if we have any NaN or Inf
    has_invalid = not np.all(np.isfinite(float_data))
    print("Any NaN/Inf in data?:", has_invalid)
    if has_invalid:
        # fix them
        float_data = np.nan_to_num(float_data, nan=0.0, posinf=1e5, neginf=-1e5)

    # 2) If the submodule is hooking shape [1,4,768], confirm that or reshape if needed:
    # For example:
    # float_data = float_data.reshape((1,4,768))

    # Let's do an ONNX check
    session = onnxruntime.InferenceSession(lora_path)
    onnx_input = {"input_x": float_data}
    onnx_output = session.run(None, onnx_input)
    print("ONNX model output shape:", onnx_output[0].shape)
    print("First few values of ONNX output:", onnx_output[0].flatten()[:5])

    # Now build the final JSON for EZKL
    input_data = {"input_data": float_data.tolist()}
    with open("input.json", "w") as f:
        json.dump(input_data, f)

    # Generate witness
    print("Generating witness...")
    start_time = time.time()
    await ezkl.gen_witness(
        data="input.json", model=lora_ezkl, output="witness.json"
    )
    end_time = time.time()
    print(f"Witness generation took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
