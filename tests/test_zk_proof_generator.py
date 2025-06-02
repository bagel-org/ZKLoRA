from __future__ import annotations

import pytest
import json
import tempfile
import os
import time
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from zklora.zk_proof_generator import (
    batch_verify_proofs,
    generate_proofs,
    ProofPaths,
    resolve_proof_paths
)


@pytest.mark.unit
class TestProofPaths:
    
    def test_proof_paths_structure(self):
        paths = ProofPaths(
            circuit="test.ezkl",
            settings="test_settings.json",
            srs="kzg.srs",
            verification_key="test.vk",
            proving_key="test.pk",
            witness="test_witness.json",
            proof="test.pf"
        )
        
        assert paths.circuit == "test.ezkl"
        assert paths.settings == "test_settings.json"
        assert paths.srs == "kzg.srs"
        assert paths.verification_key == "test.vk"
        assert paths.proving_key == "test.pk"
        assert paths.witness == "test_witness.json"
        assert paths.proof == "test.pf"


@pytest.mark.unit
def test_resolve_proof_paths():
    paths = resolve_proof_paths("/proof/dir", "test_module")
    
    assert paths.circuit == "/proof/dir/test_module.ezkl"
    assert paths.settings == "/proof/dir/test_module_settings.json"
    assert paths.srs == "/proof/dir/kzg.srs"
    assert paths.verification_key == "/proof/dir/test_module.vk"
    assert paths.proving_key == "/proof/dir/test_module.pk"
    assert paths.witness == "/proof/dir/test_module_witness.json"
    assert paths.proof == "/proof/dir/test_module.pf"


@pytest.mark.unit
class TestBatchVerifyProofs:
    
    @patch('glob.glob')
    @patch('os.path.exists')
    def test_batch_verify_no_files(self, mock_exists, mock_glob):
        mock_glob.return_value = []
        
        total_time, num_proofs = batch_verify_proofs("/dir", verbose=False)
        
        assert total_time == 0.0
        assert num_proofs == 0
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator.ezkl')
    def test_batch_verify_missing_files(self, mock_ezkl, mock_exists, mock_glob):
        mock_glob.return_value = ["/dir/test.pf"]
        mock_exists.return_value = False
        
        # Mock verify to avoid file system errors
        mock_ezkl.verify.side_effect = RuntimeError("No such file or directory")
        
        with pytest.raises(RuntimeError):
            batch_verify_proofs("/dir", verbose=False)
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator.ezkl')
    def test_batch_verify_success(self, mock_ezkl, mock_exists, mock_glob):
        mock_glob.return_value = ["/dir/test.pf"]
        mock_exists.return_value = True
        mock_ezkl.verify.return_value = True
        
        total_time, num_proofs = batch_verify_proofs("/dir", verbose=True)
        
        assert isinstance(total_time, float)
        assert total_time > 0
        assert num_proofs == 1
        mock_ezkl.verify.assert_called_once()
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator.ezkl')
    def test_batch_verify_failure(self, mock_ezkl, mock_exists, mock_glob):
        mock_glob.return_value = ["/dir/test.pf"]
        mock_exists.return_value = True
        mock_ezkl.verify.return_value = False
        
        total_time, num_proofs = batch_verify_proofs("/dir", verbose=True)
        
        assert isinstance(total_time, float)
        assert total_time > 0
        assert num_proofs == 1
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator.ezkl')
    def test_batch_verify_exception(self, mock_ezkl, mock_exists, mock_glob):
        mock_glob.return_value = ["/dir/test.pf"]
        mock_exists.return_value = True
        mock_ezkl.verify.side_effect = Exception("Verification error")
        
        # Should not raise exception, just continue processing
        with pytest.raises(Exception):
            batch_verify_proofs("/dir", verbose=True)


@pytest.mark.unit
def test_batch_verify_multiple_files():
    with patch('glob.glob') as mock_glob, \
         patch('os.path.exists', return_value=True), \
         patch('zklora.zk_proof_generator.ezkl') as mock_ezkl:
        
        mock_glob.return_value = [
            "/dir/test1.pf",
            "/dir/test2.pf",
            "/dir/test3.pf"
        ]
        mock_ezkl.verify.return_value = True
        
        total_time, num_proofs = batch_verify_proofs("/dir", verbose=False)
        
        assert isinstance(total_time, float)
        assert total_time > 0
        assert num_proofs == 3


@pytest.mark.unit
class TestGenerateProofs:
    
    @patch('glob.glob')
    @patch('os.makedirs')
    def test_generate_proofs_no_onnx_files(self, mock_makedirs, mock_glob):
        mock_glob.return_value = []
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output"))
        
        assert result is None
        mock_makedirs.assert_called_once_with("/output", exist_ok=True)
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    def test_generate_proofs_no_matching_json(self, mock_isfile, mock_makedirs, mock_glob):
        mock_glob.return_value = ["/onnx/test.onnx"]
        mock_isfile.return_value = False
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output", verbose=True))
        
        assert result == (0, 0, 0, 0, 0)
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    @patch('onnx.load')
    @patch('builtins.open')
    @patch('json.load')
    @patch('onnxruntime.InferenceSession')
    @patch('zklora.zk_proof_generator.ezkl')
    @patch('time.time')
    @patch('os.remove')
    def test_generate_proofs_success(self, mock_remove, mock_time, mock_ezkl, 
                                   mock_onnx_session, mock_json_load, mock_open,
                                   mock_onnx_load, mock_isfile, mock_makedirs, mock_glob):
        # Setup mocks
        mock_glob.return_value = ["/onnx/test.onnx"]
        mock_isfile.return_value = True
        
        # Mock ONNX model
        mock_param = Mock()
        mock_param.dims = [10, 10]
        mock_onnx_model = Mock()
        mock_onnx_model.graph.initializer = [mock_param]
        mock_onnx_load.return_value = mock_onnx_model
        
        # Mock JSON data
        mock_json_load.return_value = {"input_data": [[1, 2, 3]]}
        
        # Mock ONNX runtime
        mock_session = Mock()
        mock_session.run.return_value = [Mock(shape=(1, 3))]
        mock_onnx_session.return_value = mock_session
        
        # Mock ezkl operations
        mock_ezkl.PyRunArgs.return_value = Mock(
            input_visibility="public",
            output_visibility="public", 
            param_visibility="private",
            logrows=20
        )
        mock_ezkl.gen_settings = Mock()
        mock_ezkl.compile_circuit = Mock()
        mock_ezkl.gen_srs = Mock()
        mock_ezkl.setup = Mock()
        mock_ezkl.gen_witness = AsyncMock()
        mock_ezkl.mock.return_value = True
        mock_ezkl.prove.return_value = True
        
        # Mock time progression
        mock_time.side_effect = [0, 1, 2, 3, 4, 5]  # Each operation takes 1 second
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output", verbose=True))
        
        assert result is not None
        total_settings, total_witness, total_prove, total_params, count_files = result
        assert isinstance(total_settings, (int, float))
        assert isinstance(total_witness, (int, float))
        assert isinstance(total_prove, (int, float))
        assert total_params == 100  # 10 * 10 parameters
        assert count_files == 1
        
        # Verify ezkl calls
        mock_ezkl.gen_settings.assert_called_once()
        mock_ezkl.compile_circuit.assert_called_once()
        mock_ezkl.setup.assert_called_once()
        mock_ezkl.gen_witness.assert_called_once()
        mock_ezkl.prove.assert_called_once()
        mock_remove.assert_called_once()  # Remove proving key
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    @patch('onnx.load')
    @patch('builtins.open')
    @patch('json.load')
    @patch('onnxruntime.InferenceSession')
    @patch('zklora.zk_proof_generator.ezkl')
    @patch('time.time')
    def test_generate_proofs_witness_failure(self, mock_time, mock_ezkl, 
                                           mock_onnx_session, mock_json_load, mock_open,
                                           mock_onnx_load, mock_isfile, mock_makedirs, mock_glob):
        # Setup mocks
        mock_glob.return_value = ["/onnx/test.onnx"]
        mock_isfile.return_value = True
        
        # Mock ONNX model
        mock_param = Mock()
        mock_param.dims = [5, 5]
        mock_onnx_model = Mock()
        mock_onnx_model.graph.initializer = [mock_param]
        mock_onnx_load.return_value = mock_onnx_model
        
        # Mock JSON data
        mock_json_load.return_value = {"input_data": [[1, 2, 3]]}
        
        # Mock ONNX runtime
        mock_session = Mock()
        mock_session.run.return_value = [Mock(shape=(1, 3))]
        mock_onnx_session.return_value = mock_session
        
        # Mock ezkl operations - witness generation fails
        mock_ezkl.PyRunArgs.return_value = Mock(logrows=20)
        mock_ezkl.gen_settings = Mock()
        mock_ezkl.compile_circuit = Mock()
        mock_ezkl.gen_srs = Mock()
        mock_ezkl.setup = Mock()
        mock_ezkl.gen_witness = AsyncMock(side_effect=RuntimeError("Witness gen failed"))
        
        mock_time.side_effect = [0, 1, 2, 3]  # Setup time tracking
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output"))
        
        assert result == (1, 0, 0, 25, 0)  # Settings time but no witness/prove time
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    @patch('onnx.load')
    @patch('builtins.open')
    @patch('json.load')
    @patch('onnxruntime.InferenceSession')
    @patch('zklora.zk_proof_generator.ezkl')
    @patch('time.time')
    def test_generate_proofs_mock_failure(self, mock_time, mock_ezkl, 
                                        mock_onnx_session, mock_json_load, mock_open,
                                        mock_onnx_load, mock_isfile, mock_makedirs, mock_glob):
        # Setup similar to success test but mock fails
        mock_glob.return_value = ["/onnx/test.onnx"]
        mock_isfile.return_value = True
        
        mock_param = Mock()
        mock_param.dims = [3, 3]
        mock_onnx_model = Mock()
        mock_onnx_model.graph.initializer = [mock_param]
        mock_onnx_load.return_value = mock_onnx_model
        
        mock_json_load.return_value = {"input_data": [[1, 2, 3]]}
        
        mock_session = Mock()
        mock_session.run.return_value = [Mock(shape=(1, 3))]
        mock_onnx_session.return_value = mock_session
        
        # Mock ezkl operations - mock test fails
        mock_ezkl.PyRunArgs.return_value = Mock(logrows=20)
        mock_ezkl.gen_settings = Mock()
        mock_ezkl.compile_circuit = Mock()
        mock_ezkl.gen_srs = Mock()
        mock_ezkl.setup = Mock()
        mock_ezkl.gen_witness = AsyncMock()
        mock_ezkl.mock.return_value = False  # Mock fails
        
        mock_time.side_effect = [0, 1, 2, 3, 4]
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output"))
        
        assert result == (1, 0, 0, 9, 0)  # Settings time but no witness time since mock fails
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    @patch('onnx.load')
    @patch('builtins.open')
    @patch('json.load')
    @patch('onnxruntime.InferenceSession')
    @patch('zklora.zk_proof_generator.ezkl')
    @patch('time.time')
    def test_generate_proofs_prove_failure(self, mock_time, mock_ezkl, 
                                         mock_onnx_session, mock_json_load, mock_open,
                                         mock_onnx_load, mock_isfile, mock_makedirs, mock_glob):
        # Setup similar to success test but prove fails
        mock_glob.return_value = ["/onnx/test.onnx"]
        mock_isfile.return_value = True
        
        mock_param = Mock()
        mock_param.dims = [2, 2]
        mock_onnx_model = Mock()
        mock_onnx_model.graph.initializer = [mock_param]
        mock_onnx_load.return_value = mock_onnx_model
        
        mock_json_load.return_value = {"input_data": [[1, 2, 3]]}
        
        mock_session = Mock()
        mock_session.run.return_value = [Mock(shape=(1, 3))]
        mock_onnx_session.return_value = mock_session
        
        # Mock ezkl operations - prove fails
        mock_ezkl.PyRunArgs.return_value = Mock(logrows=20)
        mock_ezkl.gen_settings = Mock()
        mock_ezkl.compile_circuit = Mock()
        mock_ezkl.gen_srs = Mock()
        mock_ezkl.setup = Mock()
        mock_ezkl.gen_witness = AsyncMock()
        mock_ezkl.mock.return_value = True
        mock_ezkl.prove.return_value = False  # Prove fails
        
        mock_time.side_effect = [0, 1, 2, 3, 4, 5, 6]
        
        result = asyncio.run(generate_proofs("/onnx", "/json", "/output"))
        
        assert result == (1, 1, 1, 4, 0)  # All times but no successful files
    
    @patch('glob.glob')
    @patch('os.makedirs')
    @patch('os.path.isfile')
    def test_generate_proofs_existing_srs(self, mock_isfile, mock_makedirs, mock_glob):
        """Test that existing SRS file is not regenerated"""
        mock_glob.return_value = ["/onnx/test.onnx"] 
        # Return True for JSON file and SRS file exists
        mock_isfile.side_effect = lambda path: "srs" in path or ".json" in path
        
        with patch('onnx.load') as mock_onnx_load, \
             patch('builtins.open'), \
             patch('json.load') as mock_json_load, \
             patch('onnxruntime.InferenceSession') as mock_onnx_session, \
             patch('zklora.zk_proof_generator.ezkl') as mock_ezkl, \
             patch('time.time') as mock_time:
            
            # Setup basic mocks
            mock_param = Mock()
            mock_param.dims = [1, 1]
            mock_onnx_model = Mock()
            mock_onnx_model.graph.initializer = [mock_param]
            mock_onnx_load.return_value = mock_onnx_model
            
            mock_json_load.return_value = {"input_data": [[1]]}
            
            mock_session = Mock()
            mock_session.run.return_value = [Mock(shape=(1, 1))]
            mock_onnx_session.return_value = mock_session
            
            mock_ezkl.PyRunArgs.return_value = Mock(logrows=20)
            mock_ezkl.gen_settings = Mock()
            mock_ezkl.compile_circuit = Mock()
            mock_ezkl.gen_srs = Mock()  # Should not be called
            mock_ezkl.setup = Mock()
            mock_ezkl.gen_witness = AsyncMock()
            mock_ezkl.mock.return_value = True
            mock_ezkl.prove.return_value = True
            
            mock_time.side_effect = [0, 1, 2, 3, 4, 5]
            
            with patch('os.remove'):
                result = asyncio.run(generate_proofs("/onnx", "/json", "/output"))
            
            # gen_srs should not be called since SRS file exists
            mock_ezkl.gen_srs.assert_not_called()
            assert result == (1, 1, 1, 1, 1) 