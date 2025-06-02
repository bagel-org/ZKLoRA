from __future__ import annotations

import pytest
import torch
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock

from zklora.mpi_lora_onnx_exporter import (
    normalize_lora_matrices_mpi,
    LoraShapeTransformerMPI,
    export_lora_onnx_json_mpi,
    export_lora_onnx_json_mpi_optimized
)


@pytest.mark.unit
class TestNormalizeLoraMatricesMpi:
    
    def test_normalize_correct_shapes(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(768, 16)
        B = torch.randn(16, 768)
        
        A_norm, B_norm, in_dim, r, out_dim = normalize_lora_matrices_mpi(
            A, B, x_data
        )
        
        assert A_norm.shape == (768, 16)
        assert B_norm.shape == (16, 768)
        assert in_dim == 768
        assert r == 16
        assert out_dim == 768
    
    def test_normalize_transposed_A(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(16, 768)
        B = torch.randn(16, 768)
        
        A_norm, B_norm, in_dim, r, out_dim = normalize_lora_matrices_mpi(
            A, B, x_data
        )
        
        assert A_norm.shape == (768, 16)
        assert B_norm.shape == (16, 768)
        assert torch.allclose(A_norm, A.transpose(0, 1))
    
    def test_normalize_transposed_B(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(768, 16)
        B = torch.randn(768, 16)
        
        A_norm, B_norm, in_dim, r, out_dim = normalize_lora_matrices_mpi(
            A, B, x_data
        )
        
        assert A_norm.shape == (768, 16)
        assert B_norm.shape == (16, 768)
        assert torch.allclose(B_norm, B.transpose(0, 1))
    
    def test_normalize_both_transposed(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(16, 768)
        B = torch.randn(768, 16)
        
        A_norm, B_norm, in_dim, r, out_dim = normalize_lora_matrices_mpi(
            A, B, x_data
        )
        
        assert A_norm.shape == (768, 16)
        assert B_norm.shape == (16, 768)
    
    def test_normalize_invalid_A_shape(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(100, 200)
        B = torch.randn(16, 768)
        
        with pytest.raises(ValueError, match="A shape"):
            normalize_lora_matrices_mpi(A, B, x_data)
    
    def test_normalize_invalid_B_shape(self):
        x_data = np.random.randn(1, 128, 768)
        A = torch.randn(768, 16)
        B = torch.randn(100, 200)
        
        with pytest.raises(ValueError, match="B shape"):
            normalize_lora_matrices_mpi(A, B, x_data)


@pytest.mark.unit
class TestLoraShapeTransformerMPI:
    
    def test_initialization(self):
        A = torch.randn(768, 16)
        B = torch.randn(16, 768)
        transformer = LoraShapeTransformerMPI(A, B, 1, 128, 768)
        
        assert transformer.batch_size == 1
        assert transformer.seq_len == 128
        assert transformer.hidden_dim == 768
        assert hasattr(transformer, 'A')
        assert hasattr(transformer, 'B')
    
    def test_forward_pass(self):
        batch_size = 2
        seq_len = 64
        hidden_dim = 512
        rank = 8
        
        A = torch.randn(hidden_dim, rank)
        B = torch.randn(rank, hidden_dim)
        transformer = LoraShapeTransformerMPI(
            A, B, batch_size, seq_len, hidden_dim
        )
        
        x_1d = torch.randn(1, batch_size * seq_len * hidden_dim)
        output = transformer.forward(x_1d)
        
        assert output.shape == (1, batch_size * seq_len * hidden_dim)
    
    def test_forward_computation(self):
        A = torch.eye(4)
        B = torch.eye(4) * 2
        transformer = LoraShapeTransformerMPI(A, B, 1, 1, 4)
        
        x = torch.ones(1, 4)
        output = transformer.forward(x)
        
        expected_base = (x.view(1, 1, 4) @ A) @ B
        x_mean = x.view(1, 1, 4).mean()
        A_sum = A.sum()
        B_sum = B.sum()
        expected = expected_base + x_mean + A_sum + B_sum
        
        assert torch.allclose(
            output, expected.view(1, -1), atol=1e-6
        )


@pytest.mark.unit
class TestExportLoraOnnxJsonMpi:
    
    @patch('torch.onnx.export')
    def test_export_basic(self, mock_onnx_export):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_data = np.random.randn(1, 128, 768)
            
            submodule = Mock()
            lora_A_module = Mock()
            lora_B_module = Mock()
            lora_A_module.weight = torch.randn(768, 16)
            lora_B_module.weight = torch.randn(16, 768)
            
            submodule.lora_A = {'default': lora_A_module}
            submodule.lora_B = {'default': lora_B_module}
            
            export_lora_onnx_json_mpi(
                "test_module",
                x_data,
                submodule,
                tmpdir,
                verbose=True
            )
            
            mock_onnx_export.assert_called_once()
            
            json_path = os.path.join(tmpdir, "test_module.json")
            assert os.path.exists(json_path)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            assert 'input_data' in data
    
    def test_export_no_lora_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_data = np.random.randn(1, 128, 768)
            submodule = Mock()
            submodule.lora_A = None
            submodule.lora_B = None
            
            export_lora_onnx_json_mpi(
                "test_module",
                x_data,
                submodule,
                tmpdir,
                verbose=True
            )
            
            files = os.listdir(tmpdir)
            assert len(files) == 0
    
    @patch('torch.onnx.export')
    def test_export_shape_error(self, mock_onnx_export):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_data = np.random.randn(1, 128, 768)
            
            submodule = Mock()
            lora_A_module = Mock()
            lora_B_module = Mock()
            lora_A_module.weight = torch.randn(100, 200)
            lora_B_module.weight = torch.randn(300, 400)
            
            submodule.lora_A = {'default': lora_A_module}
            submodule.lora_B = {'default': lora_B_module}
            
            export_lora_onnx_json_mpi(
                "test_module",
                x_data,
                submodule,
                tmpdir,
                verbose=True
            )
            
            mock_onnx_export.assert_not_called()


@pytest.mark.unit
class TestExportLoraOnnxJsonMpiOptimized:
    
    @patch('zklora.low_rank_circuit.export_optimized_lora_circuit')
    def test_optimized_export(self, mock_export):
        mock_export.return_value = {'status': 'success'}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            x_data = np.random.randn(1, 128, 768)
            base_activations = np.random.randn(1, 128, 768)
            
            submodule = Mock()
            lora_A_module = Mock()
            lora_B_module = Mock()
            lora_A_module.weight = torch.randn(768, 16)
            lora_B_module.weight = torch.randn(16, 768)
            
            submodule.lora_A = {'default': lora_A_module}
            submodule.lora_B = {'default': lora_B_module}
            
            result = export_lora_onnx_json_mpi_optimized(
                "test_module",
                x_data,
                submodule,
                base_activations,
                tmpdir,
                use_optimization=True,
                verbose=True
            )
            
            mock_export.assert_called_once()
            assert result == {'status': 'success'}
    
    @patch('zklora.mpi_lora_onnx_exporter.export_lora_onnx_json_mpi')
    def test_fallback_to_non_optimized(self, mock_export):
        with tempfile.TemporaryDirectory() as tmpdir:
            x_data = np.random.randn(1, 128, 768)
            base_activations = np.random.randn(1, 128, 768)
            
            submodule = Mock()
            
            export_lora_onnx_json_mpi_optimized(
                "test_module",
                x_data,
                submodule,
                base_activations,
                tmpdir,
                use_optimization=False,
                verbose=True
            )
            
            mock_export.assert_called_once_with(
                "test_module", x_data, submodule, tmpdir, True
            )


@pytest.mark.unit
def test_special_character_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        x_data = np.random.randn(1, 10, 32)
        
        submodule = Mock()
        lora_A_module = Mock()
        lora_B_module = Mock()
        lora_A_module.weight = torch.randn(32, 4)
        lora_B_module.weight = torch.randn(4, 32)
        
        submodule.lora_A = {'default': lora_A_module}
        submodule.lora_B = {'default': lora_B_module}
        
        with patch('torch.onnx.export'):
            export_lora_onnx_json_mpi(
                "transformer.h.0.attn/c_attn",
                x_data,
                submodule,
                tmpdir,
                verbose=False
            )
        
        json_path = os.path.join(
            tmpdir, "transformer_h_0_attn_c_attn.json"
        )
        assert os.path.exists(json_path) 