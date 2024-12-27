import ezkl
import asyncio

async def run_ezkl_proof():
    model_onnx = "lora_onnx_params/base_model_model_transformer_h_2_attn_c_attn.onnx"
    compiled_path = "model.ezkl"
    input_data = "intermediate_activations/base_model_model_transformer_h_2_attn_c_attn.json"
    # e.g. ...
    ezkl.gen_settings(model_onnx)
    ezkl.compile_circuit(model_onnx, compiled_path, "settings.json")
    ezkl.gen_srs("kzg.srs", 20)
    ezkl.setup(compiled_path, "vk.key", "pk.key", "kzg.srs")

    await ezkl.gen_witness(
        data=input_data,
        model=compiled_path,
        output="witness.json"
    )

if __name__=="__main__":
    asyncio.run(run_ezkl_proof())
