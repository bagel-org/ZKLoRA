from zklora import export_lora_submodules_flattened
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
lora_model = PeftModel.from_pretrained(base_model, "q1e123/peft-starcoder-lora-a100")
lora_model.eval()
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

export_lora_submodules_flattened(
    model=lora_model,
    tokenizer=tokenizer,
    input_text="Hello from LoRA",
    output_dir="lora_onnx_params",
    json_dir="intermediate_activations",
    submodule_key=None   # or e.g. "attn.c_attn" if you only want that submodule
)
