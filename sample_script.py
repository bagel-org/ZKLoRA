from zklora import export_lora_submodules, generate_proofs_async, verify_proof_batch

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    # 1) Load base & LoRA model
    base_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    lora_model = PeftModel.from_pretrained(base_model, "palsp/gpt2-lora")
    lora_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

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
        json_dir="intermediate_activations",
        submodule_key="attn.c_attn",
    )

    # 3) Generate proofs for each ONNX+JSON pair
    # We'll store circuit artifacts, keys, proofs, etc., in "proof_artifacts".
    # Only the submodules named 'attn.c_attn' will appear in lora_onnx_params + intermediate_activations
    # because that's what we filtered in step 2.
    total_settings_time, total_witness_time, total_prove_time, count_onnx_files = (
        asyncio.run(
            generate_proofs_async(
                onnx_dir="lora_onnx_params",
                json_dir="intermediate_activations",
                output_dir="proof_artifacts",
            )
        )
    )

    verify_proof_batch("lora_onnx_params", "proof_artifacts")


if __name__ == "__main__":
    main()
