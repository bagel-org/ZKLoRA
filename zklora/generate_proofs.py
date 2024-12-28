import os
import glob
import onnx
import ezkl
import time
import numpy as np
import json
import onnxruntime
import asyncio


async def generate_proofs_async(
    onnx_dir: str,
    json_dir: str,
    output_dir: str = ".",
    do_verify: bool = True
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
    """

    os.makedirs(output_dir, exist_ok=True)

    onnx_files = glob.glob(os.path.join(onnx_dir, "*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {onnx_dir}.")
        return

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

        # We'll define consistent filenames
        circuit_name   = os.path.join(output_dir, f"{base_name}.ezkl")       # compiled circuit
        settings_file  = os.path.join(output_dir, f"{base_name}_settings.json")
        srs_file       = os.path.join(output_dir, "kzg.srs")
        vk_file        = os.path.join(output_dir, f"{base_name}.vk")
        pk_file        = os.path.join(output_dir, f"{base_name}.pk")
        witness_file   = os.path.join(output_dir, f"{base_name}_witness.json")
        proof_file     = os.path.join(output_dir, f"{base_name}.pf")

        py_args = ezkl.PyRunArgs()
        py_args.input_visibility = "public"
        py_args.output_visibility = "public"
        py_args.param_visibility = "fixed"
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
                data=json_path,
                model=circuit_name,
                output=witness_file
            )
        except RuntimeError as e:
            print(f"Failed to generate witness: {e}")
            continue

        if not ezkl.mock(witness_file, circuit_name):
            print("Mock run failed, skipping.")
            continue

        end_time = time.time()
        print(f"Witness gen took {end_time - start_time:.2f} sec")

        # 4) prove
        print("Generating proof...")
        prove_ok = ezkl.prove(
            witness_file,
            circuit_name,
            pk_file,
            proof_file,
            "single",
            srs_file
        )
        print("Proof result:", prove_ok)
        if not prove_ok:
            print(f"Proof generation failed for {base_name}")
            continue

        # optional verify
        if do_verify:
            print("Verifying proof...")
            verify_ok = ezkl.verify(proof_file, settings_file, vk_file, srs_file)
            print("Verification result:", verify_ok)
            if verify_ok:
                print("Proof verified successfully!")
            else:
                print("Verification failed.")

        print(f"Done with {base_name}.\n")


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
    out_dir  = "proof_artifacts"

    # Run everything in one single event loop
    asyncio.run(generate_proofs_async(
        onnx_dir=onnx_dir,
        json_dir=json_dir,
        output_dir=out_dir,
        do_verify=True
    ))
