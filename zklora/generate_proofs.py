import os
import glob
import onnx
import ezkl
import time
import numpy as np
import json
import onnxruntime
import asyncio

def get_filenames(proof_dir: str, base_name: str):
    """
    Retrieves paths for all required proof-related files given a directory and base name.

    Args:
        proof_dir (str): Directory containing the proof artifacts
        base_name (str): Base name of the files (without extension)

    Returns:
        tuple | None: A 7-tuple containing paths to:
            - circuit file (.ezkl)
            - settings file (_settings.json)
            - SRS file (kzg.srs)
            - verification key file (.vk)
            - proving key file (.pk)
            - witness file (_witness.json)
            - proof file (.pf)
            Returns None if any required file is missing.
    """
    circuit_name = os.path.join(proof_dir, f"{base_name}.ezkl")
    settings_file = os.path.join(proof_dir, f"{base_name}_settings.json")
    srs_file = os.path.join(proof_dir, "kzg.srs")
    vk_file = os.path.join(proof_dir, f"{base_name}.vk")
    pk_file = os.path.join(proof_dir, f"{base_name}.pk")
    witness_file = os.path.join(proof_dir, f"{base_name}_witness.json")
    proof_file = os.path.join(proof_dir, f"{base_name}.pf")

    return (
        circuit_name,
        settings_file,
        srs_file,
        vk_file,
        pk_file,
        witness_file,
        proof_file,
    )


def verify_proof_batch(onnx_dir: str, proof_dir: str) -> None:
    """
    Batch verifies proofs for all ONNX models in the specified directory.

    Args:
        onnx_dir (str): Directory containing ONNX model files
        proof_dir (str): Directory containing proof artifacts (proofs, verification keys, etc.)

    Returns:
        tuple[float, int]: Total time spent verifying proofs, number of proofs verified
    """
    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}.")
        return

    for onnx_path in onnx_files:
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        names = get_filenames(proof_dir, base_name)
        if names is None:
            continue
        (
            circuit_name,
            settings_file,
            srs_file,
            vk_file,
            pk_file,
            witness_file,
            proof_file,
        ) = names
        print(f"Verifying proof for {base_name}...")
        start_time = time.time()
        verify_ok = ezkl.verify(proof_file, settings_file, vk_file, srs_file)
        end_time = time.time()
        print(f"Verification took {end_time - start_time:.2f} seconds")
        if verify_ok:
            print(f"Proof verified successfully for {base_name}!\n")
        else:
            print(f"Verification failed for {base_name}.\n")
    
    return end_time - start_time, len(onnx_files)


async def generate_proofs_async(
    onnx_dir: str,
    json_dir: str,
    output_dir: str = "."
):
    """
    Asynchronously scans onnx_dir for .onnx files and json_dir for .json files.
    For each matching pair, runs:
      1) gen_settings + compile_circuit
      2) gen_srs + setup
      3) gen_witness (async)
      4) prove + optional verify

    Since this function is fully async, you can call it once with:
      asyncio.run(generate_proofs_async(...))
    without hitting "no running event loop" in a loop.

    ## Args:
        onnx_dir (str): Directory containing ONNX model files
        json_dir (str): Directory containing input JSON files
        output_dir (str): Directory to store proof artifacts (default: current directory)

    ## Returns:
        - total_settings_time (float): Total time spent on settings/setup
        - total_witness_time (float): Total time spent generating witnesses
        - total_prove_time (float): Total time spent generating proofs
        - count_onnx_files (int): Number of ONNX files successfully processed
    """

    os.makedirs(output_dir, exist_ok=True)

    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}.")
        return

    total_settings_time = 0
    total_witness_time = 0
    total_prove_time = 0
    count_onnx_files = 0
    total_params = 0
    for onnx_path in onnx_files:
        base_name = os.path.splitext(os.path.basename(onnx_path))[0]
        json_path = os.path.join(json_dir, base_name + ".json")
        if not os.path.isfile(json_path):
            print(f"No matching JSON for {onnx_path}, skipping.")
            continue

        print("==========================================")
        print(f"Preparing to prove with ONNX: {onnx_path}")
        print(f"Matching JSON: {json_path}")

        onnx_model = onnx.load(onnx_path)
        param_count = sum(np.prod(param.dims) for param in onnx_model.graph.initializer)
        print(f"Number of parameters: {param_count:,}")
        total_params += param_count

        names = get_filenames(output_dir, base_name)
        if names is None:
            continue
        (
            circuit_name,
            settings_file,
            srs_file,
            vk_file,
            pk_file,
            witness_file,
            proof_file,
        ) = names

        py_args = ezkl.PyRunArgs()
        py_args.input_visibility = "public"
        py_args.output_visibility = "public"
        py_args.param_visibility = "private"
        py_args.logrows = 20

        print("Generating settings & compiling circuit...")
        start_time = time.time()

        # 1) gen_settings + compile_circuit
        ezkl.gen_settings(onnx_path, settings_file, py_run_args=py_args)
        ezkl.compile_circuit(onnx_path, circuit_name, settings_file)

        # 2) SRS + setup
        if not os.path.isfile(srs_file):
            ezkl.gen_srs(srs_file, py_args.logrows)
        ezkl.setup(circuit_name, vk_file, pk_file, srs_file)
        end_time = time.time()
        print(f"Setup for {base_name} took {end_time - start_time:.2f} sec")
        total_settings_time += end_time - start_time

        # Local check
        with open(json_path, "r") as f:
            data = json.load(f)
        input_array = np.array(data["input_data"], dtype=np.float32)
        print("Input shape from JSON:", input_array.shape)
        session = onnxruntime.InferenceSession(onnx_path)
        out = session.run(None, {"input_x": input_array})
        print("Local ONNX output shape:", out[0].shape)

        # 3) gen_witness (async)
        print("Generating witness (async)...")
        start_time = time.time()
        try:
            await ezkl.gen_witness(
                data=json_path, model=circuit_name, output=witness_file
            )
        except RuntimeError as e:
            print(f"Failed to generate witness: {e}")
            continue

        if not ezkl.mock(witness_file, circuit_name):
            print("Mock run failed, skipping.")
            continue

        end_time = time.time()
        print(f"Witness gen took {end_time - start_time:.2f} sec")
        total_witness_time += end_time - start_time
        # 4) prove
        print("Generating proof...")
        start_time = time.time()
        prove_ok = ezkl.prove(
            witness_file, circuit_name, pk_file, proof_file, "single", srs_file
        )
        end_time = time.time()
        print(f"Proof gen took {end_time - start_time:.2f} sec")
        total_prove_time += end_time - start_time

        if not prove_ok:
            print(f"Proof generation failed for {base_name}")
            continue

        print(f"Done with {base_name}.\n")
        count_onnx_files += 1

    return total_settings_time, total_witness_time, total_prove_time, total_params, count_onnx_files


if __name__ == "__main__":
    """
    Example top-level usage:
    1) Flatten submodules as usual
    2) Call this script via: python generate_proofs_async.py
    """
    # Example usage:
    # from zklora import export_lora_submodules_flattened
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from peft import PeftModel
    #
    # base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    # lora_model = PeftModel.from_pretrained(base_model, "q1e123/peft-starcoder-lora-a100")
    # lora_model.eval()
    #
    # tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # export_lora_submodules_flattened(... "attn.c_attn"...)
    #
    # Now run the proofs:
    # asyncio.run(generate_proofs_async(
    #     onnx_dir="lora_onnx_params",
    #     json_dir="intermediate_activations",
    #     output_dir="proof_artifacts",
    #     do_verify=True
    # ))

    import sys
    import asyncio

    # or parse from sys.argv
    onnx_dir = "lora_onnx_params"
    json_dir = "intermediate_activations"
    out_dir = "proof_artifacts"

    # Run everything in one single event loop
    asyncio.run(
        generate_proofs_async(
            onnx_dir=onnx_dir, json_dir=json_dir, output_dir=out_dir, do_verify=True
        )
    )
