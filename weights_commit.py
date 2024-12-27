import json
import struct
import numpy as np
import csv
import subprocess


def flatten_list(nested_list):
    """
    Recursively flattens a nested list structure into a single-level list.
    If the input is not a list, returns it as a single-element list.
    Empty lists return empty lists. For nested lists, only the first element
    is processed recursively.

    Args:
        nested_list: A potentially nested list structure

    Returns:
        A flattened list containing non-list elements
    """
    if not isinstance(nested_list, list):
        return [nested_list]
    if len(nested_list) == 0:
        return []
    if isinstance(nested_list[0], list):
        return flatten_list(nested_list[0])
    return nested_list


def float32_to_uint64(activations):
    """
    Converts a list of float32 values to uint64 by reinterpreting the bytes.
    Each float32 (4 bytes) is padded with 4 zero bytes to create uint64 (8 bytes).

    Args:
        activations: List of float32 values

    Returns:
        List of uint64 values representing the same bit patterns as the input floats
    """
    float32_array = np.array(activations, dtype=np.float32)
    uint64_encoded = []

    for float_val in float32_array:
        # Pack float32 into bytes then interpret as uint64
        bytes_val = struct.pack("f", float_val)
        uint64_val = struct.unpack("Q", bytes_val + b"\x00" * 4)[0]  # Pad to 8 bytes
        uint64_encoded.append(uint64_val)

    return uint64_encoded


def generate_merkle_tree(activations_csv_path):
    """
    Generates a Merkle tree from the given CSV file using lambdaworks CLI,
    writes the root hash to a file and returns the root hash.

    Args:
        activations_csv_path: Path to the CSV file containing the activations

    Returns:
        str: The Merkle tree root hash

    Raises:
        subprocess.CalledProcessError: If the Merkle tree generation fails
    """
    # Generate Merkle tree using lambdaworks CLI
    try:
        result = subprocess.run(
            [
                "cargo",
                "run",
                "--manifest-path",
                "libs/lambdaworks/Cargo.toml",
                "--release",
                "--bin",
                "merkle-tree-cli",
                "generate-tree",
                activations_csv_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Error generating Merkle tree: {e}")
        print("Standard output:", e.stdout)
        print("Standard error:", e.stderr)
        raise

    # Extract root from stdout using split on ':'
    root_line = [line for line in result.stdout.split("\n") if "root:" in line][0]
    root = root_line.split(":")[1].strip().replace('"', "")

    # Write root to file
    with open("root.txt", "w") as f:
        f.write(root)

    return root


if __name__ == "__main__":

    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    with open(activations_path, "r") as f:
        activations = json.load(f)

    activations = flatten_list(activations["input_data"])

    uint64_activations = float32_to_uint64(activations)

    hex_activations = ["0x" + hex(val)[2:].upper() for val in uint64_activations]

    activations_csv_path = "activations.csv"

    # Write without using csv.writer to have full control over formatting
    with open(activations_csv_path, "w", newline="") as f:
        f.write(";".join(hex_activations))

    # Generate tree and get root
    generate_merkle_tree(activations_csv_path)
