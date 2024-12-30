from zklora import export_lora_submodules_flattened, generate_proofs_async, verify_proof

import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    # 1) Load base & LoRA model
    base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    lora_model = PeftModel.from_pretrained(
        base_model, "q1e123/peft-starcoder-lora-a100"
    )
    lora_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

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
        submodule_key="attn.c_attn"
        input_text="Hello from LoRA",
        output_dir="lora_onnx_params",  # ONNX output
        json_dir="intermediate_activations",  # JSON output
        submodule_key="attn.c_attn",  # filter for submodules named 'attn.c_attn'
    )

    # 3) Generate proofs for each ONNX+JSON pair
    # We'll store circuit artifacts, keys, proofs, etc., in "proof_artifacts".
    # Only the submodules named 'attn.c_attn' will appear in lora_onnx_params + intermediate_activations
    # because that's what we filtered in step 2.
    asyncio.run(
        generate_proofs_async(
            onnx_dir="lora_onnx_params",
            json_dir="intermediate_activations",
            output_dir="proof_artifacts",
        )
    )

    # verify_proof(proof_file, settings_file, vk_file, srs_file)


if __name__ == "__main__":
    #main()
    verify_proof(
        "proof_artifacts/base_model_model_transformer_h_0_attn_c_attn.pf",
        "proof_artifacts/base_model_model_transformer_h_0_attn_c_attn_settings.json",
        "proof_artifacts/base_model_model_transformer_h_0_attn_c_attn.vk",
        "proof_artifacts/kzg.srs",
    )
