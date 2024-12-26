import onnx
import ezkl
import time
import numpy as np
import asyncio
import json

def flatten_list(nested_list):
    if not isinstance(nested_list, list):
        return [nested_list]
    if len(nested_list) == 0:
        return []
    if isinstance(nested_list[0], list):
        return flatten_list(nested_list[0])
    return nested_list

async def main():

    lora_path = "lora_onnx_params/base_model_model_lm_head.onnx"
    lora_ezkl = "model.ezkl"
    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    
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
    # ezkl.calibrate_settings("mnist_mlp.onnx", "settings.json", target="resources")
    ezkl.compile_circuit(lora_path, lora_ezkl, "settings.json")
    ezkl.gen_srs("kzg.srs", py_run_args.logrows)
    ezkl.setup(lora_ezkl, "vk.key", "pk.key", "kzg.srs")
    end_time = time.time()
    print(f"Setup took {end_time - start_time:.2f} seconds")

    #
    # Prove
    #

    # Read and process the activations file
    with open(activations_path, "r") as f:
        activations_data = json.load(f)
    
    print(activations_data)
    # print the shape of the input data
    print(np.array(activations_data["input_data"][0]).shape)
    # print the fields of activations_data
    print(activations_data.keys())
    
    # Try running the activations through the ONNX model to verify
    import onnxruntime
    
    # Create ONNX inference session
    session = onnxruntime.InferenceSession(lora_path)
    
    # Reshape the input data to match the expected 3D shape
    input_data = np.array(activations_data["input_data"], dtype=np.float32)
    #input_data = input_data.squeeze()  # Remove extra dimensions
    # OR if you need specific dimensions:
    input_data = input_data.reshape(1, 4, 768)  # Adjust numbers based on your model's requirements

    onnx_input = {"input_x": input_data}
    
    # Run ONNX inference
    onnx_output = session.run(None, onnx_input)
    print("ONNX model output shape:", onnx_output[0].shape)
    print("First few values of ONNX output:", onnx_output[0].flatten()[:5])
    
    # Flatten each value in activations_data
    # flattened_values = [flatten_list(value) for value in activations_data.values()]
    # activations_data = {"input_data": flattened_values}
    
    # Modify how the data is written to match EZKL's expected format
    input_data = {
        "input_data": input_data.tolist()  # Flatten the nested structure
    }
    
    # Write to file
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

