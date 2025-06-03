"""Tests for the Halo2 wrapper."""
import json
import numpy as np
import pytest
from pathlib import Path
import sys
import unittest
import zklora_halo2

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
    
    assert "input" in witness
    assert "weight_a" in witness
    assert "weight_b" in witness
    assert "output" in witness
    
    # Check shapes
    assert np.array(witness["input"]).shape == input_data.shape
    assert np.array(witness["weight_a"]).shape == weight_a.shape
    assert np.array(witness["weight_b"]).shape == weight_b.shape
    
    # Check scaling
    scale = settings["scale"]
    np.testing.assert_array_almost_equal(
        np.array(witness["input"]) / scale,
        input_data
    )

def test_mock(prover, test_data):
    """Test mock verification."""
    input_data, weight_a, weight_b = test_data
    # Ensure prover has some settings before gen_witness
    prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    witness = prover.gen_witness(input_data, weight_a, weight_b)
    
    # Call mock without passing settings
    assert prover.mock(witness) is True 
    
    # Test with invalid shapes (still without passing settings to mock directly)
    invalid_witness = witness.copy()
    invalid_witness["output"] = np.zeros((3, 3)).tolist()
    assert prover.mock(invalid_witness) is False

    # Call mock WITH settings
    mock_settings = prover.settings.copy()
    mock_settings["scale"] = 5e5 # Arbitrary change to make it a new settings dict
    assert prover.mock(witness, settings=mock_settings) is True
    assert prover.settings == mock_settings # Check settings updated

@pytest.mark.asyncio
async def test_prove_verify(prover, test_data, tmp_path):
    """Test proof generation and verification."""
    input_data, weight_a, weight_b = test_data
    # Initial witness generation (relies on prover.settings from gen_settings call if any, or default)
    # Ensure settings are on the prover instance before generating witness if prove/verify don't set them initially
    initial_settings = prover.gen_settings(
        input_shape=input_data.shape,
        output_shape=(input_data.shape[0], weight_b.shape[1])
    )
    # prover.settings is now initial_settings

    witness = prover.gen_witness(input_data, weight_a, weight_b)
    
    proof_path = tmp_path / "proof1.bin"
    # Call prove without passing settings (uses existing self.settings)
    result = await prover.prove(witness, proof_path) 
    assert result is True
    assert proof_path.exists()
    
    # Call verify without passing settings (uses existing self.settings)
    verify_result = await prover.verify(proof_path) 
    assert verify_result is True

    # Now test passing settings directly to prove and verify
    new_settings = prover.gen_settings(
        input_shape=input_data.shape, 
        output_shape=(input_data.shape[0], weight_b.shape[1]), 
        scale=2e4 # Different scale
    )
    # Witness should ideally be regenerated if settings affecting it (like scale) change
    # However, the prove/verify methods are being tested for their settings override path,
    # not necessarily for end-to-end correctness with overridden-on-the-fly settings for witness generation.
    # For covering the `if settings is not None:` line, the content of witness is secondary to settings being passed.
    
    proof_path_new_settings = tmp_path / "proof2.bin"
    # Call prove WITH settings
    result_new_settings = await prover.prove(witness, proof_path_new_settings, settings=new_settings)
    assert result_new_settings is True
    assert prover.settings == new_settings # Check settings were updated on the instance

    # Call verify WITH settings (prover.settings is already new_settings from the prove call)
    # To specifically test verify's settings override, we could reset prover.settings or use another instance
    # Or, pass a *different* new_settings to verify
    verify_settings_override = new_settings.copy()
    verify_settings_override["scale"] = 3e4 # Make it distinct
    
    verify_result_new_settings = await prover.verify(proof_path_new_settings, settings=verify_settings_override)
    assert verify_result_new_settings is True
    assert prover.settings == verify_settings_override # Check settings were updated by verify

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
    actual_output = np.array(witness["output"]) / scale
    
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
    invalid_witness["output"] = np.zeros((3, 3)).tolist()
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

class TestZKLoRAHalo2(unittest.TestCase):
    def test_proof_generation_and_verification(self):
        input_data = [1.0, 2.0]
        weight_a = [3.0, 4.0]
        weight_b = [5.0, 6.0]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)  # Check dummy proof size

        result = zklora_halo2.verify_proof(proof, [1.0, 2.0])
        self.assertTrue(result)

    def test_empty_inputs(self):
        proof = zklora_halo2.generate_proof([], [], [])
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, [])
        self.assertTrue(result)

    def test_large_inputs(self):
        input_data = [float(i) for i in range(100)]
        weight_a = [float(i) for i in range(100, 200)]
        weight_b = [float(i) for i in range(200, 300)]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, input_data)
        self.assertTrue(result)

    def test_negative_inputs(self):
        input_data = [-1.0, -2.0]
        weight_a = [-3.0, -4.0]
        weight_b = [-5.0, -6.0]

        proof = zklora_halo2.generate_proof(input_data, weight_a, weight_b)
        self.assertIsInstance(proof, bytes)
        self.assertEqual(len(proof), 32)

        result = zklora_halo2.verify_proof(proof, input_data)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main() 