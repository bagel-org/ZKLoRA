import json
from typing import Iterable

import numpy as np

PRIME = 2**61 - 1
CHALLENGE = 1315423911  # fixed challenge value


def _poly_eval(coeffs: Iterable[int], x: int, mod: int) -> int:
    """Evaluates polynomial defined by `coeffs` at x modulo mod."""
    result = 0
    for c in reversed(list(coeffs)):
        result = (result * x + int(c)) % mod
    return result


def commit_activations(activations_path: str, challenge: int = CHALLENGE) -> str:
    """Return polynomial commitment of activations stored in JSON file."""
    with open(activations_path, "r") as f:
        data = json.load(f)
    arr = np.array(data["input_data"], dtype=np.int64).reshape(-1)
    val = _poly_eval(arr, challenge, PRIME)
    return hex(val)


def verify_commitment(
    activations_path: str, commitment: str, challenge: int = CHALLENGE
) -> bool:
    """Verify polynomial commitment against activations."""
    expected = commit_activations(activations_path, challenge)
    return expected == commitment.lower()

