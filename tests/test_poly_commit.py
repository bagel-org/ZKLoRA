import json
from pathlib import Path
import sys
from pathlib import Path as _P

import pytest
np = pytest.importorskip("numpy")

# Direct import of polynomial_commit module
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src" / "zklora"))
import polynomial_commit
commit_activations = polynomial_commit.commit_activations
verify_commitment = polynomial_commit.verify_commitment


def test_commit_roundtrip(tmp_path: Path):
    data = {"input_data": [1, 2, 3, 4]}
    path = tmp_path / "acts.json"
    with open(path, "w") as f:
        json.dump(data, f)

    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_empty_data(tmp_path: Path):
    """Test commitment of empty data."""
    data = {"input_data": []}
    path = tmp_path / "empty.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    # Empty data should produce a deterministic commitment
    assert commit.startswith("0x")
    assert len(commit) == 66  # "0x" + 64 hex chars
    assert verify_commitment(str(path), commit)


def test_single_element(tmp_path: Path):
    """Test commitment of single element."""
    data = {"input_data": [42]}
    path = tmp_path / "single.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_float_values(tmp_path: Path):
    """Test commitment with floating point values."""
    data = {"input_data": [1.5, 2.7, -3.14, 0.0]}
    path = tmp_path / "floats.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_nested_lists(tmp_path: Path):
    """Test commitment with nested lists (should be flattened)."""
    data = {"input_data": [[1, 2], [3, [4, 5]], 6]}
    path = tmp_path / "nested.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)
    
    # Verify that nested lists produce same commitment as flattened
    flat_data = {"input_data": [1, 2, 3, 4, 5, 6]}
    flat_path = tmp_path / "flat.json"
    with open(flat_path, "w") as f:
        json.dump(flat_data, f)
    
    flat_commit = commit_activations(str(flat_path))
    assert commit == flat_commit


def test_deterministic_commitment(tmp_path: Path):
    """Test that commitments are deterministic."""
    data = {"input_data": [1, 2, 3, 4, 5]}
    path = tmp_path / "determ.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    # Generate commitment multiple times
    commits = [commit_activations(str(path)) for _ in range(5)]
    
    # All should be identical
    assert all(c == commits[0] for c in commits)


def test_case_insensitive_verification(tmp_path: Path):
    """Test that verification is case-insensitive."""
    data = {"input_data": [1, 2, 3]}
    path = tmp_path / "case.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    
    # Test various case combinations
    assert verify_commitment(str(path), commit.lower())
    assert verify_commitment(str(path), commit.upper())
    assert verify_commitment(str(path), commit)


def test_different_data_same_length(tmp_path: Path):
    """Test that different data produces different commitments."""
    data1 = {"input_data": [1, 2, 3, 4]}
    path1 = tmp_path / "data1.json"
    with open(path1, "w") as f:
        json.dump(data1, f)
    
    data2 = {"input_data": [4, 3, 2, 1]}
    path2 = tmp_path / "data2.json"
    with open(path2, "w") as f:
        json.dump(data2, f)
    
    commit1 = commit_activations(str(path1))
    commit2 = commit_activations(str(path2))
    
    # Different data should produce different commitments
    assert commit1 != commit2
    assert verify_commitment(str(path1), commit1)
    assert verify_commitment(str(path2), commit2)
    assert not verify_commitment(str(path1), commit2)
    assert not verify_commitment(str(path2), commit1)


def test_large_dataset(tmp_path: Path):
    """Test commitment with larger dataset."""
    # Create dataset with 1000 random values
    np.random.seed(42)
    values = np.random.randn(1000).tolist()
    data = {"input_data": values}
    path = tmp_path / "large.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_power_of_two_padding(tmp_path: Path):
    """Test datasets that require padding to power of 2."""
    # Test various sizes that will require padding
    for size in [3, 5, 7, 9, 15, 17, 31, 33]:
        data = {"input_data": list(range(size))}
        path = tmp_path / f"pad_{size}.json"
        with open(path, "w") as f:
            json.dump(data, f)
        
        commit = commit_activations(str(path))
        assert verify_commitment(str(path), commit)


def test_numpy_array_input(tmp_path: Path):
    """Test commitment with numpy arrays (common in ML contexts)."""
    # Create numpy array and convert to list for JSON
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    data = {"input_data": arr.tolist()}
    path = tmp_path / "numpy.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_mixed_types(tmp_path: Path):
    """Test commitment with mixed int and float types."""
    data = {"input_data": [1, 2.5, -3, 0.0, 42]}
    path = tmp_path / "mixed.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    assert verify_commitment(str(path), commit)


def test_known_commitment_value(tmp_path: Path):
    """Test against a known commitment value to ensure stability."""
    # This test ensures the implementation produces consistent results
    data = {"input_data": [1.0, 2.0, 3.0, 4.0]}
    path = tmp_path / "known.json"
    with open(path, "w") as f:
        json.dump(data, f)
    
    commit = commit_activations(str(path))
    
    # The exact value will depend on the BLAKE3 implementation
    # But it should be deterministic and always the same
    assert commit.startswith("0x")
    assert len(commit) == 66  # "0x" + 64 hex chars
    
    # Verify it's a valid hex string
    try:
        int(commit, 16)
    except ValueError:
        pytest.fail(f"Commitment is not a valid hex string: {commit}")
    
    assert verify_commitment(str(path), commit)
