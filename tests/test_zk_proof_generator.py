"""Tests for the ZKProofGenerator."""
import json
import numpy as np
import onnx
import pytest
from pathlib import Path
from typing import Tuple
import sys
import logging
from unittest.mock import patch, AsyncMock, MagicMock

# Adjust sys.path to include the 'src' directory, parent of 'zklora' package
_P = Path
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))

from zklora.zk_proof_generator import ZKProofGenerator, generate_proofs, resolve_proof_paths

@pytest.fixture
def test_data():
    """Test data fixture."""
    input_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight_a = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32).reshape(2, 2)
    weight_b = np.array([[1.0], [1.5]], dtype=np.float32)
    return input_data, weight_a, weight_b

@pytest.fixture
def onnx_model_path(tmp_path, test_data) -> Path:
    """Create a test ONNX model."""
    input_data, weight_a, weight_b = test_data
    
    # Create ONNX model
    input_tensor = onnx.helper.make_tensor_value_info(
        'input_x', onnx.TensorProto.FLOAT, input_data.shape
    )
    output_tensor = onnx.helper.make_tensor_value_info(
        'output', onnx.TensorProto.FLOAT, (input_data.shape[0], weight_b.shape[1])
    )
    
    # Create weight initializers
    weight_a_init = onnx.helper.make_tensor(
        'weight_a', onnx.TensorProto.FLOAT, weight_a.shape, weight_a.flatten()
    )
    weight_b_init = onnx.helper.make_tensor(
        'weight_b', onnx.TensorProto.FLOAT, weight_b.shape, weight_b.flatten()
    )
    
    # Create nodes
    node1 = onnx.helper.make_node(
        'MatMul',
        ['input_x', 'weight_a'],
        ['temp']
    )
    node2 = onnx.helper.make_node(
        'MatMul',
        ['temp', 'weight_b'],
        ['output']
    )
    
    # Create graph
    graph = onnx.helper.make_graph(
        [node1, node2],
        'test_model',
        [input_tensor],
        [output_tensor],
        [weight_a_init, weight_b_init]
    )
    
    # Create model
    model = onnx.helper.make_model(graph)
    model.ir_version = 7
    model.opset_import[0].version = 13
    
    # Save model
    model_path = tmp_path / "test_model.onnx"
    onnx.save(model, str(model_path))
    
    return model_path

@pytest.fixture
def proof_generator(onnx_model_path, tmp_path) -> ZKProofGenerator:
    """Create a ZKProofGenerator instance."""
    return ZKProofGenerator(
        onnx_model_path=onnx_model_path,
        out_dir=tmp_path / "proofs"
    )

@pytest.mark.asyncio
async def test_proof_generation(proof_generator, test_data):
    """Test basic proof generation."""
    input_data, weight_a, weight_b = test_data
    
    # Mock the proof generation
    success, proof_path = await proof_generator.generate_proof(
        input_data=input_data,
        weight_a=weight_a,
        weight_b=weight_b,
        proof_id="test"
    )
    assert not success  # Mock proof should fail
    assert proof_path.exists()
    assert proof_path.name == "test.proof"

@pytest.mark.asyncio
async def test_proof_verification(proof_generator, test_data):
    """Test proof verification."""
    input_data, weight_a, weight_b = test_data
    
    # Generate mock proof
    success, proof_path = await proof_generator.generate_proof(
        input_data=input_data,
        weight_a=weight_a,
        weight_b=weight_b,
        proof_id="test_verify"
    )
    assert not success  # Mock proof should fail
    assert proof_path.exists()
    
    # Verify mock proof
    verify_success = await proof_generator.verify_proof(proof_path)
    assert not verify_success  # Mock proof should fail verification

@pytest.mark.asyncio
async def test_batch_verification(proof_generator, test_data):
    """Test batch verification of multiple proofs."""
    input_data, weight_a, weight_b = test_data
    
    # Generate multiple mock proofs
    proof_paths = []
    for i in range(3):
        success, path = await proof_generator.generate_proof(
            input_data=input_data,
            weight_a=weight_a,
            weight_b=weight_b,
            proof_id=f"batch_{i}"
        )
        assert not success  # Mock proof should fail
        assert path.exists()
        proof_paths.append(path)
    
    # Verify batch of mock proofs
    verify_success = await proof_generator.verify_proofs(proof_paths)
    assert not verify_success  # Mock proofs should fail verification

def test_shape_validation(proof_generator):
    """Test validation of incompatible shapes."""
    # Invalid input/weight_a shapes
    with pytest.raises(ValueError):
        proof_generator._validate_shapes(
            np.zeros((2, 3)),
            np.zeros((4, 4)),
            np.zeros((4, 2))
        )
    
    # Invalid weight_a/weight_b shapes
    with pytest.raises(ValueError):
        proof_generator._validate_shapes(
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            np.zeros((4, 2))
        )

def test_settings_generation(proof_generator, test_data):
    """Test settings generation and persistence."""
    input_data, weight_a, weight_b = test_data
    
    # Check that settings were generated
    assert proof_generator.settings is not None
    assert "input_shape" in proof_generator.settings
    assert "output_shape" in proof_generator.settings
    
    # Check settings file was created
    settings_file = proof_generator.out_dir / "settings.json"
    assert settings_file.exists()
    
    # Verify settings content
    with open(settings_file) as f:
        loaded_settings = json.load(f)
    assert loaded_settings == proof_generator.settings

@pytest.mark.asyncio
async def test_error_handling(proof_generator):
    """Test error handling for invalid inputs."""
    # Test with invalid shapes
    with pytest.raises(ValueError):
        await proof_generator.generate_proof(
            input_data=np.zeros((2, 3)),
            weight_a=np.zeros((4, 4)),
            weight_b=np.zeros((4, 2)),
            proof_id="error_test"
        )
    
    # Test with non-existent proof file
    with pytest.raises(FileNotFoundError):
        await proof_generator.verify_proof(Path("nonexistent.proof"))

@pytest.mark.asyncio
async def test_different_data_types(proof_generator):
    """Test handling of different input data types."""
    # Test with Python lists
    input_data = [[1.0, 2.0], [3.0, 4.0]]
    weight_a = [[0.1, 0.2], [0.3, 0.4]]
    weight_b = [[1.0], [1.5]]
    
    success, proof_path = await proof_generator.generate_proof(
        input_data=input_data,
        weight_a=weight_a,
        weight_b=weight_b,
        proof_id="list_test"
    )
    assert not success  # Mock proof should fail
    assert proof_path.exists()

@pytest.mark.asyncio
async def test_generate_proofs_no_onnx_files(tmp_path, caplog):
    onnx_dir = tmp_path / "onnx"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"
    onnx_dir.mkdir()
    json_dir.mkdir()
    caplog.set_level(logging.WARNING)

    result = await generate_proofs(onnx_dir, json_dir, output_dir)
    assert result is False
    assert f"No ONNX files found in {onnx_dir}" in caplog.text

@pytest.mark.asyncio
async def test_generate_proofs_missing_json(tmp_path, caplog):
    onnx_dir = tmp_path / "onnx"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"
    onnx_dir.mkdir()
    json_dir.mkdir()
    # Create a dummy ONNX file
    (onnx_dir / "model1.onnx").touch()
    caplog.set_level(logging.WARNING)

    # Mock ZKProofGenerator to prevent actual proof generation attempts
    with patch('zklora.zk_proof_generator.ZKProofGenerator') as MockZKGenerator:
        result = await generate_proofs(onnx_dir, json_dir, output_dir)
    
    # Even if it continues, result might be True if loop completes with no errors from *its* perspective
    # The function returns False only if no ONNX files at all. Otherwise, it processes what it can.
    assert result is True # It should complete, just log warnings for missing parts
    assert "JSON file not found for model1" in caplog.text
    MockZKGenerator.assert_not_called() # Should not attempt to init if JSON is missing

@pytest.mark.asyncio
@patch('zklora.zk_proof_generator.ZKProofGenerator')
async def test_generate_proofs_success_verbose(MockZKGenerator, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    onnx_dir = tmp_path / "onnx"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"
    onnx_dir.mkdir()
    json_dir.mkdir()

    (onnx_dir / "model1.onnx").touch()
    mock_params = {"input": [], "weight_a": [], "weight_b": []}
    with open(json_dir / "model1.json", "w") as f:
        json.dump(mock_params, f)

    mock_instance = MockZKGenerator.return_value
    mock_instance.generate_proof = AsyncMock(return_value=(True, output_dir / "model1.proof"))

    result = await generate_proofs(onnx_dir, json_dir, output_dir, verbose=True)
    assert result is True
    MockZKGenerator.assert_called_once_with(onnx_model_path=(onnx_dir / "model1.onnx"), out_dir=output_dir)
    mock_instance.generate_proof.assert_called_once_with(
        input_data=mock_params["input"],
        weight_a=mock_params["weight_a"],
        weight_b=mock_params["weight_b"],
        proof_id="model1"
    )
    assert "Generated proof for model1" in caplog.text

@pytest.mark.asyncio
@patch('zklora.zk_proof_generator.ZKProofGenerator')
async def test_generate_proofs_failure_verbose(MockZKGenerator, tmp_path, caplog):
    caplog.set_level(logging.ERROR)
    onnx_dir = tmp_path / "onnx"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"
    onnx_dir.mkdir()
    json_dir.mkdir()

    (onnx_dir / "model1.onnx").touch()
    mock_params = {"input": [], "weight_a": [], "weight_b": []}
    with open(json_dir / "model1.json", "w") as f:
        json.dump(mock_params, f)

    mock_instance = MockZKGenerator.return_value
    mock_instance.generate_proof = AsyncMock(return_value=(False, output_dir / "model1.proof")) # Simulate failure

    result = await generate_proofs(onnx_dir, json_dir, output_dir, verbose=True)
    assert result is True # Function itself completes
    assert "Failed to generate proof for model1" in caplog.text 

def test_resolve_proof_paths_no_ids(tmp_path):
    proof_dir = tmp_path / "proofs"
    proof_dir.mkdir()
    (proof_dir / "proof1.proof").touch()
    (proof_dir / "proof2.proof").touch()
    (proof_dir / "other.file").touch() # Should be ignored

    paths = resolve_proof_paths(proof_dir, proof_ids=None)
    assert len(paths) == 2
    assert proof_dir / "proof1.proof" in paths
    assert proof_dir / "proof2.proof" in paths

def test_resolve_proof_paths_with_ids(tmp_path):
    proof_dir = tmp_path / "proofs"
    proof_dir.mkdir()
    # Create dummy files, though their existence isn't strictly necessary for this function
    (proof_dir / "id_A.proof").touch()
    (proof_dir / "id_C.proof").touch()

    proof_ids_to_resolve = ["id_A", "id_B"] # id_B.proof does not exist
    paths = resolve_proof_paths(proof_dir, proof_ids=proof_ids_to_resolve)
    
    assert len(paths) == 2
    assert paths[0] == proof_dir / "id_A.proof"
    assert paths[1] == proof_dir / "id_B.proof" # Path is constructed even if file doesn't exist

def test_resolve_proof_paths_empty_dir_no_ids(tmp_path):
    proof_dir = tmp_path / "empty_proofs"
    proof_dir.mkdir()
    paths = resolve_proof_paths(proof_dir, proof_ids=None)
    assert len(paths) == 0

def test_resolve_proof_paths_empty_ids_list(tmp_path):
    proof_dir = tmp_path / "some_proofs"
    proof_dir.mkdir()
    (proof_dir / "some.proof").touch()
    paths = resolve_proof_paths(proof_dir, proof_ids=[])
    assert len(paths) == 0 

@pytest.mark.asyncio
async def test_generate_proof_mock_fails(proof_generator, test_data, caplog):
    """Test ZKProofGenerator.generate_proof when prover.mock fails."""
    input_data, weight_a, weight_b = test_data
    caplog.set_level(logging.ERROR) # Not strictly necessary for ValueError, but good for other errors

    # Mock prover.mock to return False
    proof_generator.prover.mock = MagicMock(return_value=False)

    with pytest.raises(ValueError, match="Mock verification failed"):
        await proof_generator.generate_proof(
            input_data=input_data,
            weight_a=weight_a,
            weight_b=weight_b,
            proof_id="mock_fail_test"
        )
    proof_generator.prover.mock.assert_called_once() # Ensure mock was called

@pytest.mark.asyncio
async def test_generate_proof_prover_fails(proof_generator, test_data, caplog):
    """Test ZKProofGenerator.generate_proof when prover.prove fails."""
    input_data, weight_a, weight_b = test_data
    caplog.set_level(logging.ERROR)

    # Mock prover.prove to return False
    proof_generator.prover.prove = AsyncMock(return_value=False)
    # Mock prover.mock to return True so we pass that stage
    proof_generator.prover.mock = MagicMock(return_value=True)

    success, proof_path = await proof_generator.generate_proof(
        input_data=input_data,
        weight_a=weight_a,
        weight_b=weight_b,
        proof_id="prover_fail_test"
    )

    assert success is False
    expected_proof_path = proof_generator.out_dir / "prover_fail_test.proof"
    assert proof_path == expected_proof_path
    assert f"Failed to generate proof for prover_fail_test" in caplog.text
    proof_generator.prover.prove.assert_called_once() # Ensure prove was called

def test_get_proof_path(proof_generator):
    """Test ZKProofGenerator.get_proof_path."""
    proof_id = "sample_proof_id"
    expected_path = proof_generator.out_dir / f"{proof_id}.proof"
    actual_path = proof_generator.get_proof_path(proof_id)
    assert actual_path == expected_path

def test_zk_proof_generator_init_no_settings_path(onnx_model_path, tmp_path):
    """Test ZKProofGenerator init when settings_path is None, ensuring settings are generated."""
    out_dir = tmp_path / "proofs_init_test"
    # Mock Halo2Prover and its methods that would be called during init
    with patch('zklora.zk_proof_generator.Halo2Prover') as MockHalo2Prover:
        mock_prover_instance = MockHalo2Prover.return_value
        mock_generated_settings = {"input_shape": [1,10], "output_shape": [1,5], "scale": 100}
        mock_prover_instance.gen_settings = MagicMock(return_value=mock_generated_settings)
        mock_prover_instance.save_settings = MagicMock()

        generator = ZKProofGenerator(onnx_model_path=onnx_model_path, settings_path=None, out_dir=out_dir)

        assert generator.settings == mock_generated_settings
        mock_prover_instance.gen_settings.assert_called_once_with(
            input_shape=generator.input_shape, # These are from the mock ONNX model
            output_shape=generator.output_shape
        )
        expected_settings_file = out_dir / "settings.json"
        mock_prover_instance.save_settings.assert_called_once_with(expected_settings_file) 