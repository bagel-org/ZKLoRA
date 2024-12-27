import json

def flatten_list(nested_list):
    if not isinstance(nested_list, list):
        return [nested_list]
    if len(nested_list) == 0:
        return []
    if isinstance(nested_list[0], list):
        return flatten_list(nested_list[0])
    return nested_list

if __name__ == "__main__":
    
    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    with open(activations_path, "r") as f:
        activations = json.load(f)

    activations = flatten_list(activations["input_data"])

    print(len(activations))
