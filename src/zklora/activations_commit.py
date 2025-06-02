from __future__ import annotations

from .polynomial_commit import commit_activations as activations_commitment
from .polynomial_commit import verify_commitment as activations_verify_commitment


def commit_activations(activations_path: str) -> str:
    """Return polynomial commitment of activations stored in JSON file."""
    return activations_commitment(activations_path)


def verify_commitment(activations_path: str, commitment: str) -> bool:
    """Verify a polynomial commitment against activations.""" 
    return activations_verify_commitment(activations_path, commitment)

if __name__ == "__main__":
    activations_path = "intermediate_activations/base_model_model_lm_head.json"
    merkle_root = commit_activations(activations_path)
    print("Merkle root:", merkle_root)
