import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# The LoRA adapted DistilGPT2 model
lora_model_name = "ng0-k1/distilgpt2-finetuned-es"

# Load base DistilGPT2
base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Load the LoRA weights
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

input_text = "Hello, world!"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids

# Intermediate activations dictionary
intermediate_activations = {}

def make_hook(layer_name, hidden_dim):
    def hook(module, input, output):
        # output from c_attn is [batch, seq_len, 3*hidden_dim]
        # split into q, k, v
        q, k, v = output.split(hidden_dim, dim=2)
        # store just the key activations
        intermediate_activations[layer_name] = k.detach().cpu().numpy().tolist()
    return hook

# Access the transformer layers
transformer_layers = model.base_model.model.transformer.h
hidden_dim = model.base_model.model.config.n_embd  # Typically 768 for DistilGPT2

# Register hooks on c_attn. We'll extract K from the combined QKV output.
for i, block in enumerate(transformer_layers):
    layer_name = f"layer_{i}_key_projection"
    # Register a forward hook on c_attn
    block.attn.c_attn.register_forward_hook(make_hook(layer_name, hidden_dim))

# Run a forward pass to trigger hooks
with torch.no_grad():
    _ = model(input_ids)

# Create output directory for JSON files
output_dir = "intermediate_activations"
os.makedirs(output_dir, exist_ok=True)

# Save each layer's activations in EZKL-compatible format
for layer_name, activations in intermediate_activations.items():
    # EZKL expects {"input_data": [[...]]}
    data_json = {"input_data": [activations]}
    layer_json_path = os.path.join(output_dir, f"{layer_name}.json")

    with open(layer_json_path, 'w') as f:
        json.dump(data_json, f)

print("Intermediate activations saved as JSON.")
