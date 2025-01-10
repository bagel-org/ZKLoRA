import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def compute_loss(model, tokenizer, text: str) -> float:
    """
    Compute the cross-entropy loss of 'text' using the given LoRA-PEFT model.
    """
    # Tokenize the input
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()  # the usual approach for causal LM loss

    # Forward pass with labels => model computes cross-entropy for each token
    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss  # average cross-entropy over the sequence
    return loss.item()

def main():
    # 1) Load the base model
    base_model_name = "distilgpt2"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # 2) Load the LoRA PEFT adapter
    lora_model_id = "ng0-k1/distilgpt2-finetuned-es"
    peft_model = PeftModel.from_pretrained(base_model, lora_model_id)
    peft_model.eval()

    # 3) Sample text (Spanish, given the fine-tuned es model)
    text = "funky chunky monkey"

    # 4) Compute and print the loss
    loss_value = compute_loss(peft_model, tokenizer, text)
    print(f"Loss on sample text: {loss_value:.4f}")

if __name__ == "__main__":
    main()
