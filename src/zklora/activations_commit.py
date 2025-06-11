from merkly.mtree import MerkleTree
import json
import numpy as np

def get_merkle_root(activations_path: str) -> str:
    """
    Calculate the Merkle root hash of model activations stored in a JSON file.
    
    Args:
        activations_path: Path to JSON file containing model activations under "input_data" key
        
    Returns:
        str: Hexadecimal string of the Merkle root hash
    """
    # Load the intermediate activations from JSON file
    with open(activations_path, 'r') as f:
        activations = json.load(f)

    # Convert nested data to numpy array and flatten
    flattened_np = np.array(activations["input_data"]).reshape(-1)
    
    # Get and return the Merkle root hash using merkly
    # Ensure all elements are strings for merkly, as it expects str or bytes.
    # Or, ensure your data is bytes if that's more appropriate.
    # For simplicity here, converting numbers to strings.
    str_list = [str(item) for item in flattened_np.tolist()]
    if not str_list: # Handle empty list case for MerkleTree
        # merkly.MerkleTree([]) raises error, decide how to handle empty data
        # For now, returning a placeholder or raising an error.
        # This behavior should be consistent with expected output or tested.
        return "0x" + "0"*64 # Placeholder for empty data, adjust as needed

    tree = MerkleTree(str_list)
    return "0x" + tree.root.hex() # merkly provides hex output, ensure '0x' prefix if needed

if __name__ == "__main__":
    # Example path, ensure this file exists or adjust for your testing
    activations_path = "intermediate_activations/base_model_model_lm_head.json" 
    try:
        merkle_root = get_merkle_root(activations_path)
        print("Merkle root:", merkle_root)
    except FileNotFoundError:
        print(f"Error: Activations file not found at {activations_path}")
    except KeyError:
        print(f"Error: 'input_data' key missing in {activations_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
