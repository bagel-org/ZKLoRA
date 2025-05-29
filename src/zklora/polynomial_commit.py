import json
from typing import Iterable, List, Union

from blake3 import blake3  # type: ignore

# Merkle-based vector commitment parameters
LEAF_EMPTY = b"\x00" * 32  # same as EMPTY_HASH in Rust implementation


def _hash_leaf(value: Union[int, float]) -> bytes:
    """Hash a single scalar value exactly like the Rust implementation.

    The Rust helper converts the *binary* representation of the f64 to BLAKE3.
    We replicate that here so that Python roots match Rust roots byte-for-byte.
    """
    # All activations are serialised as 8-byte big-endian just like f64::to_be_bytes
    if isinstance(value, float):
        import struct
        byte_repr = struct.pack('>d', value)  # '>d' = big-endian double (f64)
    else:
        # Treat ints as floats to match Rust f64 representation
        import struct
        byte_repr = struct.pack('>d', float(value))
    return blake3(byte_repr).digest()


def _parent_hash(left: bytes, right: bytes) -> bytes:
    """Aggregate two children into their parent node (binary tree)."""
    return blake3(left + right).digest()


def _merkle_root(values: List[Union[int, float]]) -> bytes:
    """Compute Merkle root for a list of scalar values.

    The tree is padded on the right with EMPTY leaves in order to guarantee that
    every internal node always has exactly two children, matching the behaviour
    of dusk-merkle with `Tree::<Item, H, A>::new()` where missing sub-trees are
    equal to the constant `EMPTY_SUBTREE` (32 zero bytes).
    """
    if not values:
        return LEAF_EMPTY

    # Convert to leaf hashes
    level: List[bytes] = [_hash_leaf(v) for v in values]

    # Pad to even length with EMPTY leaves
    if len(level) % 2 == 1:
        level.append(LEAF_EMPTY)

    # Build tree bottom-up until we get the root
    while len(level) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(level), 2):
            left, right = level[i], level[i + 1]
            next_level.append(_parent_hash(left, right))
        if len(next_level) % 2 == 1 and len(next_level) != 1:
            next_level.append(LEAF_EMPTY)
        level = next_level
    return level[0]


# --------------------------------------------------------------------------------------
# Public API (names preserved for backwards compatibility)
# --------------------------------------------------------------------------------------

def commit_activations(activations_path: str) -> str:
    """Return Merkle commitment of activations stored in JSON file.

    The JSON is expected to contain a key `input_data` pointing to a *flat* list
    (or nested list) of numeric scalars (int / float / numpy scalar). Nested
    lists are flattened before hashing to ensure the order is preserved.
    """
    with open(activations_path, "r") as f:
        data = json.load(f)

    # Flatten arbitrarily nested lists using numpy when available for speed
    try:
        import numpy as np  # local import to avoid hard dependency for users that do not need it

        flat_vals = np.asarray(data["input_data"], dtype=np.float64).reshape(-1).tolist()
    except Exception:
        # fallback: naïve Python flatten
        def _flatten(x):
            for y in x:
                if isinstance(y, (list, tuple)):
                    yield from _flatten(y)
                else:
                    yield y

        flat_vals = list(_flatten(data["input_data"]))

    root = _merkle_root(flat_vals)
    return "0x" + root.hex()


def verify_commitment(activations_path: str, commitment: str) -> bool:
    """Verify a Merkle commitment against activations."""
    expected = commit_activations(activations_path)
    return expected.lower() == commitment.lower()

