from __future__ import annotations

import pytest
import torch
import numpy as np
import json
import os
import tempfile

from zklora.low_rank_circuit import (
    LowRankQuantizer,
    LowRankCircuitONNX,
    LookupTableCircuit,
    FusedLookupBatcher,
    BaseModelCommitment,
    export_optimized_lora_circuit
)


@pytest.mark.unit
class TestLowRankQuantizer:
    
    def test_initialization(self):
        quantizer = LowRankQuantizer(weight_bits=4, activation_bits=8)
        assert quantizer.weight_bits == 4
        assert quantizer.activation_bits == 8
        assert quantizer.weight_min == -8
        assert quantizer.weight_max == 7
        assert quantizer.activation_max == 255
    
    def test_quantize_weights_basic(self):
        quantizer = LowRankQuantizer(weight_bits=4)
        A = torch.randn(128, 16)
        B = torch.randn(16, 64)
        
        A_q, B_q, quant_params = quantizer.quantize_weights(A, B)
        
        assert A_q.shape == A.shape
        assert B_q.shape == B.shape
        assert A_q.dtype == np.int8
        assert B_q.dtype == np.int8
        assert np.all(A_q >= -8) and np.all(A_q <= 7)
        assert np.all(B_q >= -8) and np.all(B_q <= 7)
        assert 'A_scale' in quant_params
        assert 'B_scale' in quant_params
        assert 'weight_bits' in quant_params
    
    def test_quantize_weights_zero_tensor(self):
        quantizer = LowRankQuantizer()
        A = torch.zeros(10, 5)
        B = torch.zeros(5, 10)
        
        A_q, B_q, quant_params = quantizer.quantize_weights(A, B)
        
        assert np.all(A_q == 0)
        assert np.all(B_q == 0)
        assert quant_params['A_scale'] > 0
        assert quant_params['B_scale'] > 0
    
    def test_quantize_activations(self):
        quantizer = LowRankQuantizer()
        x = torch.randn(1, 32, 768)
        
        x_q, scale = quantizer.quantize_activations(x)
        
        assert x_q.shape == x.shape
        assert x_q.dtype == np.uint8
        assert np.all(x_q >= 0) and np.all(x_q <= 255)
        assert scale > 0


@pytest.mark.unit
class TestLowRankCircuitONNX:
    
    def test_initialization(self):
        A_q = np.random.randint(-8, 8, (768, 16), dtype=np.int8)
        B_q = np.random.randint(-8, 8, (16, 768), dtype=np.int8)
        quant_params = {'A_scale': 0.1, 'B_scale': 0.1, 'weight_bits': 4}
        
        circuit = LowRankCircuitONNX(A_q, B_q, quant_params, 1, 128)
        
        assert circuit.rank == 16
        assert circuit.batch_size == 1
        assert circuit.seq_len == 128
        assert circuit.A_scale == 0.1
        assert circuit.B_scale == 0.1
    
    def test_forward_pass(self):
        hidden_dim = 768
        rank = 16
        batch_size = 1
        seq_len = 128
        
        A_q = np.random.randint(-8, 8, (hidden_dim, rank), dtype=np.int8)
        B_q = np.random.randint(-8, 8, (rank, hidden_dim), dtype=np.int8)
        quant_params = {'A_scale': 0.1, 'B_scale': 0.1, 'weight_bits': 4}
        
        circuit = LowRankCircuitONNX(A_q, B_q, quant_params, batch_size, seq_len)
        
        x = torch.randn(1, batch_size * seq_len * hidden_dim)
        y = circuit.forward(x)
        
        assert y.shape == (1, batch_size * seq_len * hidden_dim)


@pytest.mark.unit
class TestLookupTableCircuit:
    
    def test_initialization(self):
        circuit = LookupTableCircuit(weight_bits=4, activation_bits=8)
        assert circuit.weight_bits == 4
        assert circuit.activation_bits == 8
        assert len(circuit.lookup_table) == 16 * 256
    
    def test_lookup_table_correctness(self):
        circuit = LookupTableCircuit(weight_bits=4, activation_bits=8)
        
        assert circuit.lookup_table[(0, 0)] == 0
        assert circuit.lookup_table[(1, 10)] == 10
        assert circuit.lookup_table[(-1, 10)] == -10
        assert circuit.lookup_table[(7, 255)] == 7 * 255
        assert circuit.lookup_table[(-8, 100)] == -800
    
    def test_export_lookup_config(self):
        circuit = LookupTableCircuit(weight_bits=4, activation_bits=8)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = circuit.export_lookup_config(f.name)
            
        assert 'lookup_type' in config
        assert config['lookup_type'] == 'multiplication'
        assert config['table_size'] == 16 * 256
        assert len(config['entries']) == 16 * 256
        
        os.unlink(f.name)


@pytest.mark.unit
class TestFusedLookupBatcher:
    
    def test_initialization(self):
        batcher = FusedLookupBatcher(batch_size=8)
        assert batcher.batch_size == 8
    
    def test_create_batched_lookup_spec(self):
        batcher = FusedLookupBatcher(batch_size=8)
        spec = batcher.create_batched_lookup_spec(100)
        
        assert spec['num_batches'] == 13
        assert spec['batch_size'] == 8
        assert spec['total_operations'] == 100
        assert spec['lookup_strategy'] == 'vectorized'
        assert spec['row_reduction_factor'] == 8


@pytest.mark.unit
class TestBaseModelCommitment:
    
    def test_initialization(self):
        commitment = BaseModelCommitment()
        assert isinstance(commitment.commitment_cache, dict)
    
    def test_compute_base_commitment(self):
        commitment = BaseModelCommitment()
        base_activations = np.random.randn(1, 128, 768)
        
        digest = commitment.compute_base_commitment(base_activations)
        
        assert isinstance(digest, str)
        assert digest.startswith("poseidon_")
        assert len(digest) == len("poseidon_") + 16
    
    def test_create_delta_circuit_config(self):
        commitment = BaseModelCommitment()
        base_commitment = "poseidon_1234567890abcdef"
        
        config = commitment.create_delta_circuit_config(base_commitment)
        
        assert config['circuit_type'] == 'lora_delta_only'
        assert config['base_model_commitment'] == base_commitment
        assert 'public_inputs' in config
        assert 'private_inputs' in config
        assert 'constraints' in config


@pytest.mark.integration
class TestExportOptimizedLoraCircuit:
    
    def test_export_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = torch.randn(768, 16)
            B = torch.randn(16, 768)
            x_data = np.random.randn(1, 128, 768)
            base_activations = np.random.randn(1, 128, 768)
            
            config = export_optimized_lora_circuit(
                "test_module",
                A, B, x_data, base_activations,
                tmpdir,
                verbose=False
            )
            
            assert 'submodule' in config
            assert config['submodule'] == 'test_module'
            assert 'optimizations' in config
            assert config['optimizations']['rank'] == 16
            
            assert os.path.exists(os.path.join(tmpdir, "test_module_optimized.onnx"))
            assert os.path.exists(os.path.join(tmpdir, "test_module_config.json"))
            assert os.path.exists(os.path.join(tmpdir, "test_module_lookup.json"))
    
    def test_export_with_special_characters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = torch.randn(64, 8)
            B = torch.randn(8, 64)
            x_data = np.random.randn(2, 16, 64)
            base_activations = np.random.randn(2, 16, 64)
            
            config = export_optimized_lora_circuit(
                "transformer.h.0.attn.c_attn",
                A, B, x_data, base_activations,
                tmpdir,
                verbose=False
            )
            
            safe_name = "transformer_h_0_attn_c_attn"
            assert os.path.exists(os.path.join(tmpdir, f"{safe_name}_optimized.onnx"))
    
    def test_export_with_verbose(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = torch.randn(32, 4)
            B = torch.randn(4, 32)
            x_data = np.random.randn(1, 10, 32)
            base_activations = np.random.randn(1, 10, 32)
            
            config = export_optimized_lora_circuit(
                "verbose_test",
                A, B, x_data, base_activations,
                tmpdir,
                verbose=True
            )
            
            assert config['optimizations']['low_rank_factorization'] is True
            assert config['performance_gains']['low_rank_speedup'] == 512
            assert config['performance_gains']['quantization_speedup'] == 8
            assert config['performance_gains']['batching_speedup'] == 3
            assert config['performance_gains']['external_commitment_speedup'] == 40
            assert config['performance_gains']['total_speedup'] == 491520
    
    def test_export_large_matrices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            A = torch.randn(1024, 64)
            B = torch.randn(64, 1024)
            x_data = np.random.randn(2, 256, 1024)
            base_activations = np.random.randn(2, 256, 1024)
            
            config = export_optimized_lora_circuit(
                "large_test",
                A, B, x_data, base_activations,
                tmpdir,
                verbose=False
            )
            
            assert config['optimizations']['rank'] == 64
            assert 'weight_quantization' in config['optimizations']
            assert 'activation_quantization' in config['optimizations']


@pytest.mark.unit
def test_quantization_edge_cases():
    quantizer = LowRankQuantizer(weight_bits=2, activation_bits=4)
    
    assert quantizer.weight_min == -2
    assert quantizer.weight_max == 1
    assert quantizer.activation_max == 15
    
    A = torch.tensor([[100.0, -100.0]])
    B = torch.tensor([[50.0], [-50.0]])
    
    A_q, B_q, params = quantizer.quantize_weights(A, B)
    
    assert np.all(A_q >= -2) and np.all(A_q <= 1)
    assert np.all(B_q >= -2) and np.all(B_q <= 1)


@pytest.mark.unit
def test_special_quantization_cases():
    quantizer = LowRankQuantizer(weight_bits=8, activation_bits=16)
    
    # Test with extreme values
    A = torch.tensor([[1000.0, -1000.0]])
    B = torch.tensor([[0.001], [-0.001]])
    
    A_q, B_q, params = quantizer.quantize_weights(A, B)
    
    assert A_q.dtype == np.int8
    assert B_q.dtype == np.int8
    
    # Test reconstruction accuracy
    A_recon = torch.from_numpy(A_q).float() * params['A_scale']
    B_recon = torch.from_numpy(B_q).float() * params['B_scale']
    
    # Check that reconstruction is reasonably close
    assert torch.abs(A - A_recon).max() < 100.0  # Allow some quantization error


@pytest.mark.unit
def test_lookup_table_edge_cases():
    # Test with minimal quantization
    circuit = LookupTableCircuit(weight_bits=1, activation_bits=1)
    
    assert len(circuit.lookup_table) == 2 * 2  # 2^1 * 2^1
    assert circuit.lookup_table[(-1, 0)] == 0
    assert circuit.lookup_table[(-1, 1)] == -1
    assert circuit.lookup_table[(0, 0)] == 0
    assert circuit.lookup_table[(0, 1)] == 0


@pytest.mark.unit
def test_fused_lookup_batcher_edge_cases():
    batcher = FusedLookupBatcher(batch_size=1)
    
    spec = batcher.create_batched_lookup_spec(1)
    assert spec['num_batches'] == 1
    assert spec['batch_size'] == 1
    assert spec['total_operations'] == 1
    
    # Test with zero operations
    spec = batcher.create_batched_lookup_spec(0)
    assert spec['num_batches'] == 0
    assert spec['total_operations'] == 0


@pytest.mark.unit
def test_base_model_commitment_edge_cases():
    commitment = BaseModelCommitment()
    
    # Test with empty activations
    empty_activations = np.array([])
    digest = commitment.compute_base_commitment(empty_activations)
    assert isinstance(digest, str)
    assert digest.startswith("poseidon_")
    
    # Test with single value
    single_value = np.array([1.0])
    digest = commitment.compute_base_commitment(single_value)
    assert isinstance(digest, str)
    
    # Test repeated calls with same data give same result
    test_data = np.array([1, 2, 3])
    digest1 = commitment.compute_base_commitment(test_data)
    digest2 = commitment.compute_base_commitment(test_data)
    assert digest1 == digest2


@pytest.mark.unit 
def test_circuit_onnx_edge_cases():
    # Test with minimal dimensions
    A_q = np.array([[1]], dtype=np.int8)
    B_q = np.array([[2]], dtype=np.int8)
    quant_params = {'A_scale': 0.5, 'B_scale': 0.25, 'weight_bits': 1}
    
    circuit = LowRankCircuitONNX(A_q, B_q, quant_params, 1, 1)
    
    assert circuit.rank == 1
    
    x = torch.tensor([[1.0]])
    y = circuit.forward(x)
    
    assert y.shape == (1, 1)


@pytest.mark.unit
def test_quantizer_activation_edge_cases():
    quantizer = LowRankQuantizer(activation_bits=1)
    
    # Test with very small values
    x = torch.tensor([[[0.001, -0.001, 0.0]]])
    x_q, scale = quantizer.quantize_activations(x)
    
    assert x_q.dtype == np.uint8
    assert np.all(x_q >= 0) and np.all(x_q <= 1)  # 2^1 - 1 = 1
    assert scale > 0 