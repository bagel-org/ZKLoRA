from __future__ import annotations

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock, PropertyMock
import asyncio

from zklora.zk_proof_generator_optimized import (
    create_optimized_chip_for_model,
    OptimizedProofPaths,
    resolve_optimized_proof_paths,
    generate_optimized_proof_single,
    generate_proofs_optimized_parallel,
    batch_verify_proofs_optimized
)


@pytest.mark.unit
class TestCreateOptimizedChipForModel:
    
    def test_create_basic_chip(self):
        model_config = {
            'optimizations': {
                'rank': 16,
                'weight_quantization': {'weight_bits': 4}
            }
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chip_path = create_optimized_chip_for_model(model_config, tmpdir)
            
            assert os.path.exists(chip_path)
            assert chip_path.endswith('chip_config.json')
            
            with open(chip_path, 'r') as f:
                chip_config = json.load(f)
            
            assert chip_config['type'] == 'low_rank_optimized'
            assert chip_config['rank'] == 16
            assert chip_config['lookup_tables_enabled'] is True
            assert chip_config['batched_lookups'] is True
    
    def test_create_chip_missing_config(self):
        model_config = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            chip_path = create_optimized_chip_for_model(model_config, tmpdir)
            
            with open(chip_path, 'r') as f:
                chip_config = json.load(f)
            
            assert chip_config['rank'] == 4


@pytest.mark.unit
class TestOptimizedProofPaths:
    
    def test_proof_paths_structure(self):
        paths = OptimizedProofPaths(
            circuit="test.ezkl",
            settings="test_settings.json",
            srs="kzg.srs",
            verification_key="test.vk",
            proving_key="test.pk",
            witness="test_witness.json",
            proof="test.pf",
            chip_config="test_chip.json",
            lookup_config="test_lookup.json"
        )
        
        assert paths.circuit == "test.ezkl"
        assert paths.settings == "test_settings.json"
        assert paths.srs == "kzg.srs"
        assert paths.verification_key == "test.vk"
        assert paths.proving_key == "test.pk"
        assert paths.witness == "test_witness.json"
        assert paths.proof == "test.pf"
        assert paths.chip_config == "test_chip.json"
        assert paths.lookup_config == "test_lookup.json"


@pytest.mark.unit
def test_resolve_optimized_proof_paths():
    paths = resolve_optimized_proof_paths("/proof/dir", "test_module")
    
    assert paths.circuit == "/proof/dir/test_module.ezkl"
    assert paths.settings == "/proof/dir/test_module_settings.json"
    assert paths.srs == "/proof/dir/kzg.srs"
    assert paths.verification_key == "/proof/dir/test_module.vk"
    assert paths.proving_key == "/proof/dir/test_module.pk"
    assert paths.witness == "/proof/dir/test_module_witness.json"
    assert paths.proof == "/proof/dir/test_module.pf"
    assert paths.chip_config == "/proof/dir/test_module_chip.json"
    assert paths.lookup_config == "/proof/dir/test_module_lookup.json"


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateOptimizedProofSingle:
    
    @patch('zklora.zk_proof_generator_optimized.ezkl')
    @patch('os.path.isfile')
    @patch('os.remove')
    async def test_generate_proof_success(
        self, mock_remove, mock_isfile, mock_ezkl
    ):
        mock_isfile.return_value = False
        mock_ezkl.prove.return_value = True
        mock_ezkl.gen_witness = AsyncMock()
        
        # Create a simpler PyRunArgs mock
        mock_py_args = type('MockPyRunArgs', (), {
            'input_visibility': 'public',
            'output_visibility': 'public', 
            'param_visibility': 'private',
            'logrows': 16,
            '__dict__': {
                'input_visibility': 'public',
                'output_visibility': 'public',
                'param_visibility': 'private',
                'logrows': 16
            }
        })()
        mock_ezkl.PyRunArgs.return_value = mock_py_args
        
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "test.onnx")
            json_path = os.path.join(tmpdir, "test.json")
            config_path = os.path.join(tmpdir, "test_config.json")
            
            config = {
                'optimizations': {
                    'rank': 16,
                    'weight_quantization': {'weight_bits': 4}
                },
                'performance_gains': {'total_speedup': 1000}
            }
            
            with open(config_path, 'w') as f:
                json.dump(config, f)
            
            result = await generate_optimized_proof_single(
                onnx_path, json_path, config_path, tmpdir, verbose=False
            )
            
            assert result['module'] == 'test'
            assert result['success'] is True
            assert result['optimization_speedup'] == 1000
            assert 'settings_time' in result
            assert 'witness_time' in result
            assert 'prove_time' in result
            assert 'total_time' in result
    
    @patch('zklora.zk_proof_generator_optimized.ezkl')
    async def test_generate_proof_no_names(self, mock_ezkl):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the config file that was missing
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, 'w') as f:
                json.dump({
                    'optimizations': {'rank': 4},
                    'performance_gains': {'total_speedup': 100}
                }, f)
            
            # Mock PyRunArgs
            mock_py_args = type('MockPyRunArgs', (), {
                'input_visibility': 'public',
                'output_visibility': 'public', 
                'param_visibility': 'private',
                'logrows': 14,
                '__dict__': {
                    'input_visibility': 'public',
                    'output_visibility': 'public',
                    'param_visibility': 'private',
                    'logrows': 14
                }
            })()
            mock_ezkl.PyRunArgs.return_value = mock_py_args
            
            # Mock other ezkl functions
            mock_ezkl.compile_circuit = Mock()
            mock_ezkl.gen_srs = Mock()
            mock_ezkl.setup = Mock()
            mock_ezkl.gen_witness = AsyncMock()
            mock_ezkl.prove.return_value = True
            
            with patch('os.path.isfile', return_value=False):
                with patch('os.remove'):
                    result = await generate_optimized_proof_single(
                        "test.onnx", "test.json", config_path, tmpdir
                    )
            
            # Expecting successful result now
            assert result is not None
            assert result['success'] is True


@pytest.mark.unit
@pytest.mark.asyncio
class TestGenerateProofsOptimizedParallel:
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator_optimized.generate_optimized_proof_single')
    async def test_parallel_generation(
        self, mock_single, mock_exists, mock_glob
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_glob.return_value = [
                f"{tmpdir}/module1_optimized.onnx",
                f"{tmpdir}/module2_optimized.onnx"
            ]
            mock_exists.return_value = True
            
            async def mock_proof_gen(*args):
                return {
                    'module': 'test',
                    'success': True,
                    'total_time': 1.0,
                    'optimization_speedup': 500
                }
            
            mock_single.side_effect = mock_proof_gen
            
            result = await generate_proofs_optimized_parallel(
                onnx_dir=tmpdir,
                json_dir=tmpdir,
                output_dir=tmpdir,
                verbose=False
            )
            
            assert result['total_modules'] == 2
            assert result['successful_proofs'] == 2
            assert 'total_time' in result
            assert 'parallel_speedup' in result
            assert 'average_proof_time' in result
            assert 'average_theoretical_speedup' in result
    
    @patch('glob.glob')
    async def test_no_onnx_files(self, mock_glob):
        mock_glob.return_value = []
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await generate_proofs_optimized_parallel(
                onnx_dir=tmpdir,
                json_dir=tmpdir,
                output_dir=tmpdir,
                verbose=False
            )
            
            assert result is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestBatchVerifyProofsOptimized:
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('builtins.__import__')
    async def test_verify_all_valid(self, mock_import, mock_exists, mock_glob):
        mock_glob.return_value = [
            "/dir/module1_optimized.pf",
            "/dir/module2_optimized.pf"
        ]
        # Mock all file checks to return True
        mock_exists.return_value = True
        
        # Create a mock ezkl module
        mock_ezkl = Mock()
        mock_ezkl.verify = AsyncMock(return_value=True)
        
        # Mock the import to return our mock ezkl
        def side_effect(name, *args, **kwargs):
            if name == 'ezkl':
                return mock_ezkl
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        results = await batch_verify_proofs_optimized(
            proof_dir="/dir", verbose=False
        )
        
        assert len(results) == 2
        assert results['module1'] is True
        assert results['module2'] is True
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('builtins.__import__')
    async def test_verify_mixed_results(self, mock_import, mock_exists, mock_glob):
        mock_glob.return_value = [
            "/dir/module1_optimized.pf",
            "/dir/module2_optimized.pf"
        ]
        
        # Mock all file checks to return True
        mock_exists.return_value = True
        
        # Create mock ezkl with conditional verify
        mock_ezkl = Mock()
        
        async def mock_verify(*args, **kwargs):
            proof_path = kwargs.get('proof_path', args[0] if args else '')
            return 'module1' in proof_path
        
        mock_ezkl.verify = mock_verify
        
        # Mock the import to return our mock ezkl
        def side_effect(name, *args, **kwargs):
            if name == 'ezkl':
                return mock_ezkl
            return __import__(name, *args, **kwargs)
        mock_import.side_effect = side_effect
        
        results = await batch_verify_proofs_optimized(
            proof_dir="/dir", verbose=False
        )
        
        assert len(results) == 2
        assert results['module1'] is True
        assert results['module2'] is False
    
    @patch('glob.glob')
    async def test_no_proof_files(self, mock_glob):
        mock_glob.return_value = []
        
        results = await batch_verify_proofs_optimized(
            proof_dir="/dir", verbose=False
        )
        
        assert len(results) == 0
    
    @patch('glob.glob')
    @patch('os.path.exists')
    @patch('zklora.zk_proof_generator_optimized.ezkl')
    async def test_verify_with_exception(
        self, mock_ezkl, mock_exists, mock_glob
    ):
        mock_glob.return_value = ["/dir/module1_optimized.pf"]
        mock_exists.return_value = True
        mock_ezkl.verify = AsyncMock(side_effect=Exception("Verify error"))
        
        results = await batch_verify_proofs_optimized(
            proof_dir="/dir", verbose=True
        )
        
        assert results['module1'] is False 