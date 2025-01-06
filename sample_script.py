from zklora import export_lora_submodules, generate_proofs, batch_verify_proofs

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
import csv


def main():
    base_model_name = "distilgpt2"
    #base_model_name = "openai-community/gpt2"
    lora_model_name = "q1e123/peft-starcoder-lora-a100"
    #lora_model_name = "palsp/gpt2-lora"
    # 1) Load base & LoRA model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    lora_model = PeftModel.from_pretrained(
        base_model, lora_model_name
    )
    lora_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # multiple text
    texts = ["Hello from LoRA", "And another test", "One more line..."]
    # 2) Export only attn.c_attn submodules
    # This will produce ONNX + JSON in the specified dirs,
    # hooking only the submodules whose name contains "attn.c_attn".
    export_lora_submodules(
        model=lora_model,
        tokenizer=tokenizer,
        input_texts=texts,  # pass list of strings
        output_dir="lora_onnx_params",
        json_dir="intermediate_activations"
    )

    csv_path = "proof_metrics.csv"
    columns = [
        "base_model", "lora", "number_of_loras", "total_params", "avg_params",
        "total_settings", "total_witness", "total_prove", 
        "avg_settings", "avg_witness", "avg_prove",
        "total_verify", "avg_verify"
    ]

    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # 3) Generate proofs for each ONNX+JSON pair
    # We'll store circuit artifacts, keys, proofs, etc., in "proof_artifacts".
    # Only the submodules named 'attn.c_attn' will appear in lora_onnx_params + intermediate_activations
    # because that's what we filtered in step 2.
    total_settings_time, total_witness_time, total_prove_time, total_params, count_onnx_files = (
        asyncio.run(
            generate_proofs(
                onnx_dir="lora_onnx_params",
                json_dir="intermediate_activations",
                output_dir="proof_artifacts",
            )
        )
    )

    total_verify_time, count_proofs = batch_verify_proofs(
        "lora_onnx_params", "proof_artifacts"
    )

    # Calculate averages
    avg_settings = total_settings_time / count_onnx_files if count_onnx_files > 0 else 0
    avg_witness = total_witness_time / count_onnx_files if count_onnx_files > 0 else 0 
    avg_prove = total_prove_time / count_onnx_files if count_onnx_files > 0 else 0
    avg_verify = total_verify_time / count_proofs if count_proofs > 0 else 0
    avg_params = total_params / count_onnx_files if count_onnx_files > 0 else 0

    # Write results to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            base_model_name,  # base_model 
            lora_model_name,  # lora
            count_onnx_files,  # number_of_loras
            total_params,  # total_params
            avg_params,  # avg_params
            total_settings_time,  # total_settings
            total_witness_time,  # total_witness  
            total_prove_time,  # total_prove
            avg_settings,  # avg_settings
            avg_witness,  # avg_witness
            avg_prove,  # avg_prove
            total_verify_time,  # total_verify
            avg_verify  # avg_verify
        ])
    


if __name__ == "__main__":
    main()
