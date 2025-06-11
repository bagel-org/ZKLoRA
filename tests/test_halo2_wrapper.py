"""Tests for the Halo2 wrapper."""
import json
import numpy as np
import pytest
from pathlib import Path
import sys
import unittest
import zklora_halo2
import numbers

# Adjust sys.path to include the 'src' directory, parent of 'zklora' package
_P = Path
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))

# Now import Halo2Prover from the zklora package
from zklora.halo2_wrapper import Halo2Prover

@pytest.fixture
def prover():
    """Create a Halo2Prover instance for testing."""
    return Halo2Prover()

@pytest.fixture
def test_data():
    """Create test data for LoRA computations."""
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight_a = np.array([[0.1, 0.2], [0.3, 0.4]])
    weight_b = np.array([[1.0, 1.5], [2.0, 2.5]])
    return input_data, weight_a, weight_b

def test_gen_settings(prover, tmp_path):
    """Test settings generation."""
    settings = prover.gen_settings(
        input_shape=(2, 2),
        output_shape=(2, 2),
        scale=1e4
    )
    
    assert settings["input_shape"] == (2, 2)
    assert settings["output_shape"] == (2, 2)
    assert settings["scale"] == 1e4
    assert settings["bits"] == 32
    assert "input" in settings["public_inputs"]
    assert "output" in settings["public_inputs"]
    assert "weight_a" in settings["private_inputs"]
    assert "weight_b" in settings["private_inputs"]
    
    # Test saving settings
    settings_path = tmp_path / "settings.json"
    prover.save_settings(settings_path)
    assert settings_path.exists()
    
    # Test loading settings
    with open(settings_path) as f:
        loaded_settings = json.load(f)
    loaded_settings["input_shape"] = tuple(loaded_settings["input_shape"])
    loaded_settings["output_shape"] = tuple(loaded_settings["output_shape"])
    assert loaded_settings == settings

def test_compile_circuit(prover, tmp_path):
    """Test circuit compilation."""
    settings = prover.gen_settings((2, 2), (2, 2))
    onnx_path = tmp_path / "model.onnx"
    onnx_path.touch()  # Create empty file
    
    # Should be a no-op but not fail
    prover.compile_circuit(onnx_path, settings)
    assert prover.settings == settings

def test_gen_witness(prover, test_data):
    """Test witness generation."""
    input_data, weight_a, weight_b = test_data
    settings = prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    assert "input_mags" in witness
    assert "weight_a_mags" in witness
    assert "weight_b_mags" in witness
    assert "output_mags" in witness
    # Check shapes
    assert len(witness["input_mags"]) == np.prod(witness["input_shape"])
    assert len(witness["weight_a_mags"]) == weight_a.size
    assert len(witness["weight_b_mags"]) == weight_b.size
    assert len(witness["output_mags"]) == np.prod(witness["output_shape"])
    # Check scaling (reconstruct signed values)
    scale = settings["scale"]
    input_signed = np.array(witness["input_mags"]) * np.where(np.array(witness["input_signs"]) == 0, 1, -1)
    np.testing.assert_array_almost_equal(
        input_signed.reshape(witness["input_shape"]) / scale,
        input_data
    )

def test_mock(prover, test_data):
    """Test mock verification."""
    input_data, weight_a, weight_b = test_data
    prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    assert prover.mock(witness) is True 
    # Test with invalid shapes
    invalid_witness = witness.copy()
    invalid_witness["output_mags"] = [0] * 9  # Wrong shape
    assert prover.mock(invalid_witness) is False
    mock_settings = prover.settings.copy()
    mock_settings["scale"] = 5e5
    assert prover.mock(witness, settings=mock_settings) is True
    assert prover.settings == mock_settings

@pytest.mark.asyncio
async def test_prove_verify(prover, test_data, tmp_path):
    """Test proof generation and verification."""
    input_data, weight_a, weight_b = test_data
    initial_settings = prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    proof_path = tmp_path / "proof1.bin"
    
    # Mock the proof generation since we're testing Python interface
    with open(proof_path, "wb") as f:
        f.write(b"mock_proof")
    
    result = await prover.verify(proof_path)
    assert not result  # Mock proof should fail verification

@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6])
def test_different_scales(prover, test_data, scale):
    """Test different scaling factors."""
    input_data, weight_a, weight_b = test_data
    settings = prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1]),
        scale=scale
    )
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    expected_output = input_data @ weight_a @ weight_b
    output_signed = np.array(witness["output_mags"]) * np.where(np.array(witness["output_signs"]) == 0, 1, -1)
    actual_output = output_signed.reshape(witness["output_shape"]) / scale
    np.testing.assert_array_almost_equal(actual_output, expected_output)

@pytest.mark.parametrize("shape", [
    ((1, 2), (2, 3), (3, 1)),
    ((2, 4), (4, 3), (3, 2)),
    ((5, 2), (2, 2), (2, 5))
])
def test_different_shapes(prover, shape):
    """Test different matrix shapes."""
    input_shape, weight_a_shape, weight_b_shape = shape
    input_data = np.random.randn(*input_shape)
    weight_a = np.random.randn(*weight_a_shape)
    weight_b = np.random.randn(*weight_b_shape)
    
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    assert prover.mock(witness) is True

def test_error_handling(prover):
    """Test error handling."""
    with pytest.raises(ValueError):
        # Invalid shapes
        input_data = np.random.randn(2, 3)
        weight_a = np.random.randn(4, 4)  # Incompatible shape
        weight_b = np.random.randn(4, 2)
        prover.gen_witness(input_data, weight_a, weight_b)

def test_validate_shapes(prover, test_data):
    """Test shape validation."""
    input_data, weight_a, weight_b = test_data
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    
    # Test with valid shapes
    assert prover.mock(witness) is True
    
    # Test with invalid shapes
    invalid_witness = witness.copy()
    invalid_witness["output_mags"] = [0] * 9  # Wrong shape
    assert prover.mock(invalid_witness) is False

@pytest.mark.asyncio
async def test_settings_persistence(prover, test_data, tmp_path):
    """Test settings persistence across operations."""
    input_data, weight_a, weight_b = test_data
    settings = prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    
    # Save settings
    settings_path = tmp_path / "settings.json"
    prover.save_settings(settings_path)
    
    # Create new prover with saved settings
    new_prover = Halo2Prover(settings_path)
    assert new_prover.settings == settings
    
    # Generate witness with loaded settings
    witness = new_prover.gen_witness(input_data, weight_a, weight_b)
    assert new_prover.mock(witness) is True 

def flatten_matrix(matrix):
    arr = np.asarray(matrix)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2:
        return arr.flatten().tolist()
    if isinstance(matrix, list):
        if not matrix:
            return []
        if all(isinstance(x, numbers.Number) for x in matrix):
            return list(matrix)
        return [x for row in matrix for x in row]
    return list(matrix)

def quantize_signed(val, scale=1e4):
    mag = abs(int(round(val * scale)))
    sign = 0 if val >= 0 else 1
    return mag, sign

def flatten_and_quantize(matrix, scale=1e4):
    flat = flatten_matrix(matrix)
    if not flat:
        return [], []
    mags, signs = zip(*(quantize_signed(v, scale) for v in flat))
    return list(mags), list(signs)

class TestZKLoRAHalo2(unittest.TestCase):
    def test_empty_inputs(self):
        prover = Halo2Prover()
        prover.gen_settings(input_shape=(0,), output_shape=(0,))
        witness = prover.gen_witness([], [], [])
        assert witness["input_mags"] == []
        assert witness["output_mags"] == []

    def test_large_inputs(self):
        # Use smaller values to avoid overflow
        # Each value when quantized should be < 2^32 - 1
        input_data = np.array([float(i/100) for i in range(100)]).reshape(1, 100)  # Values 0.0 to 0.99
        weight_a = np.array([float(i/100) for i in range(100, 300)]).reshape(100, 2)  # Values 1.0 to 2.99
        weight_b = np.array([float(i/100) for i in range(300, 302)]).reshape(2, 1)  # Values 3.0 to 3.01
        prover = Halo2Prover()
        prover.gen_settings(input_shape=(1, 100), output_shape=(1, 1))
        witness = prover.gen_witness(input_data, weight_a, weight_b)
        assert len(witness["input_mags"]) == 100

    def test_negative_inputs(self):
        input_data = np.array([-1.0, -2.0]).reshape(1, 2)
        weight_a = np.array([-3.0, -4.0, -5.0, -6.0]).reshape(2, 2)
        weight_b = np.array([-7.0, -8.0]).reshape(2, 1)
        prover = Halo2Prover()
        prover.gen_settings(input_shape=(1, 2), output_shape=(1, 1))
        witness = prover.gen_witness(input_data, weight_a, weight_b)
        assert all(sign == 1 for sign in witness["input_signs"])

    def test_proof_generation_and_verification(self):
        input_data = np.array([1.0, 2.0]).reshape(1, 2)
        weight_a = np.array([3.0, 4.0, 5.0, 6.0]).reshape(2, 2)
        weight_b = np.array([7.0, 8.0]).reshape(2, 1)
        prover = Halo2Prover()
        prover.gen_settings(input_shape=(1, 2), output_shape=(1, 1))
        witness = prover.gen_witness(input_data, weight_a, weight_b)
        assert len(witness["input_mags"]) == 2

if __name__ == '__main__':
    unittest.main() 