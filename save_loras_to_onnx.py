import torch
import torch.nn as nn
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "ng0-k1/distilgpt2-finetuned-es"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base_model, model_name)
model.eval()

output_dir = "lora_onnx_params"
os.makedirs(output_dir, exist_ok=True)

class LoraMatmulModule(nn.Module):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        # Store A and B as buffers
        self.register_buffer('A', A)
        self.register_buffer('B', B)

    def forward(self, x):
        # Perform B @ A instead of A @ B
        return self.B @ self.A

def extract_lora_param(lora_obj, module_name, param_name):
    # The printouts showed lora_obj is a ModuleDict with a "default" key containing a Linear layer.
    if isinstance(lora_obj, nn.ModuleDict):
        if 'default' in lora_obj:
            default_mod = lora_obj['default']
            if isinstance(default_mod, nn.Linear):
                return default_mod.weight
            else:
                raise ValueError(f"{module_name}.{param_name}.default is not a Linear layer.")
        else:
            raise ValueError(f"{module_name}.{param_name} ModuleDict has no 'default' key.")
    else:
        raise ValueError(f"Expected {module_name}.{param_name} to be a ModuleDict, got {type(lora_obj)}")

lora_layers = []
for name, module in model.named_modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
        lora_layers.append((name, module))

if not lora_layers:
    print("No LoRA layers found.")
else:
    for i, (name, lora_module) in enumerate(lora_layers):
        A_param = extract_lora_param(lora_module.lora_A, name, 'lora_A')
        B_param = extract_lora_param(lora_module.lora_B, name, 'lora_B')

        # Detach and move to CPU
        A = A_param.detach().cpu().float()
        B = B_param.detach().cpu().float()

        matmul_mod = LoraMatmulModule(A, B).eval()

        dummy_input = torch.zeros(1, dtype=torch.float32)
        safe_name = name.replace(".", "_").replace("/", "_")
        onnx_path = os.path.join(output_dir, f"{safe_name}.onnx")

        torch.onnx.export(
            matmul_mod,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"]
        )

        print(f"Exported LoRA params for layer '{name}' to {onnx_path}")

    print("All LoRA parameter ONNX files saved.")
