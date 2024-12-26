from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Get all named parameters from the model
for name, param in model.named_parameters():
    # Print parameter name and its shape and its value
    print(f"Parameter: {name}")
    print(f"Shape: {param.shape}")
    print(f"Value: {param.data}")
    print("-" * 50)

    # Convert parameter tensor to a flat vector and store in list
    param_vector = param.data.flatten().tolist()
    #print(f"Param vector: {param_vector}")
    
    # Create a simple commitment by hashing the values
    # Note: In practice you'd want a more secure commitment scheme
    import hashlib
    
    # Convert values to bytes and hash them
    param_bytes = str(param_vector).encode('utf-8')
    commitment = hashlib.sha256(param_bytes).hexdigest()
    
    print(f"Commitment: {commitment}")
