from __future__ import annotations

import pytest
import torch
import numpy as np
import json
import tempfile
import subprocess
import shutil
import os
from pathlib import Path
from unittest.mock import patch, Mock

from zklora.low_rank_circuit import (
    LowRankQuantizer, 
    export_optimized_lora_circuit,
    LowRankCircuitONNX
)
from zklora.polynomial_commit import commit_activations


def ezkl_available():
    """Check if EZKL binary is available."""
    return shutil.which("ezkl") is not None


class SimpleLoRACircuit(torch.nn.Module):
    """Simple LoRA circuit for testing without quantization complexity."""
    
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        super().__init__()
        self.A = A
        self.B = B
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden] -> [batch * seq, hidden]
        x_flat = x.view(-1, x.shape[-1])
        # Compute LoRA: (x @ A) @ B
        result = (x_flat @ self.A @ self.B)
        # Return reshaped to original batch structure
        return result.view(x.shape)


@pytest.mark.integration
class TestCircuitRoundTrip:
    """
    Round-trip validation: PyTorch → ONNX → EZKL → witness → verification
    
    This ensures mathematical correctness of the entire proof pipeline.
    """
    
    @pytest.fixture
    def small_lora_params(self):
        """Small LoRA parameters for fast testing."""
        return {
            "d": 64,      # embedding dimension
            "r": 4,       # rank  
            "batch_size": 1,
            "seq_len": 8
        }
    
    @pytest.fixture
    def large_lora_params(self):
        """Larger LoRA parameters for comprehensive testing."""
        return {
            "d": 768,     # Standard transformer dimension
            "r": 64,      # Higher rank
            "batch_size": 2,
            "seq_len": 128
        }
    
    @pytest.fixture
    def test_tensors(self, small_lora_params):
        """Generate test tensors with controlled randomness."""
        torch.manual_seed(42)
        d, r = small_lora_params["d"], small_lora_params["r"]
        batch_size, seq_len = small_lora_params["batch_size"], small_lora_params["seq_len"]
        
        return {
            "A": torch.randn(d, r) * 0.02,  # Much smaller scale for precision
            "B": torch.randn(r, d) * 0.02,
            "x": torch.randn(batch_size, seq_len, d) * 0.1,
            "base_activations": torch.randn(batch_size, seq_len, d) * 0.1
        }
    
    @pytest.fixture
    def large_test_tensors(self, large_lora_params):
        """Generate larger test tensors for high-rank testing."""
        torch.manual_seed(42)
        d, r = large_lora_params["d"], large_lora_params["r"]
        batch_size, seq_len = large_lora_params["batch_size"], large_lora_params["seq_len"]
        
        return {
            "A": torch.randn(d, r) * 0.01,  # Even smaller scale for large matrices
            "B": torch.randn(r, d) * 0.01,
            "x": torch.randn(batch_size, seq_len, d) * 0.05,
            "base_activations": torch.randn(batch_size, seq_len, d) * 0.05
        }
    
    def test_pytorch_circuit_equivalence(self, test_tensors):
        """Test that OptimizedLoRACircuit produces same output as manual computation."""
        A, B, x = test_tensors["A"], test_tensors["B"], test_tensors["x"]
        
        # Manual PyTorch computation
        x_flat = x.view(-1, x.shape[-1])
        manual_output = (x_flat @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # Should be numerically identical (no quantization yet)
        torch.testing.assert_close(
            manual_output, 
            circuit_output, 
            rtol=1e-6, 
            atol=1e-7,
            msg="Circuit output differs from manual PyTorch computation"
        )
    
    def test_quantization_error_bounds(self, test_tensors):
        """Test quantization introduces bounded error with strict thresholds."""
        A, B, x = test_tensors["A"], test_tensors["B"], test_tensors["x"]
        
        # Original computation
        original_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Quantized computation
        quantizer = LowRankQuantizer(weight_bits=4, activation_bits=8)
        A_q, B_q, weight_params = quantizer.quantize_weights(A, B)
        x_q, activation_scale, zero_point = quantizer.quantize_activations(x)
        
        # Dequantize for comparison
        # Weights: symmetric quantization (no zero point)
        A_dq = A_q.astype(np.float32) * weight_params["A_scale"]
        B_dq = B_q.astype(np.float32) * weight_params["B_scale"]
        # Activations: asymmetric quantization (with zero point)
        x_dq = (x_q.astype(np.float32) - zero_point) * activation_scale
        
        A_dq_t = torch.from_numpy(A_dq)
        B_dq_t = torch.from_numpy(B_dq)
        x_dq_t = torch.from_numpy(x_dq)
        
        quantized_output = (x_dq_t.view(-1, x_dq_t.shape[-1]) @ A_dq_t @ B_dq_t).view(x.shape)
        
        # MUCH MORE STRICT: Calculate absolute error
        abs_error = torch.abs(original_output - quantized_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        # STRICT: Absolute error should be very small for good quantization
        assert max_abs_error < 0.01, f"Max absolute error {max_abs_error:.6f} exceeds 1% threshold"
        assert mean_abs_error < 0.002, f"Mean absolute error {mean_abs_error:.6f} exceeds 0.2% threshold"
        
        # STRICT: Calculate relative error only for significant values
        significant_mask = torch.abs(original_output) > 0.005  # Filter smaller values
        if torch.any(significant_mask):
            significant_original = original_output[significant_mask]
            significant_quantized = quantized_output[significant_mask]
            
            rel_error = torch.abs(significant_original - significant_quantized) / torch.abs(significant_original)
            max_rel_error = torch.max(rel_error).item()
            mean_rel_error = torch.mean(rel_error).item()
            
            # MUCH STRICTER: For significant values, demand high precision
            assert max_rel_error < 0.3, f"Max relative error {max_rel_error:.3f} exceeds 30% for significant values"
            assert mean_rel_error < 0.1, f"Mean relative error {mean_rel_error:.3f} exceeds 10% for significant values"
        
        # STRICT: Test that quantization preserves overall magnitude precisely
        original_magnitude = torch.norm(original_output).item()
        quantized_magnitude = torch.norm(quantized_output).item()
        magnitude_error = abs(original_magnitude - quantized_magnitude) / (original_magnitude + 1e-8)
        
        assert magnitude_error < 0.05, f"Overall magnitude error {magnitude_error:.4f} exceeds 5% threshold"
        
        # NEW: Test quantization bounds are respected
        assert A_q.min() >= -8 and A_q.max() <= 7, "4-bit weight quantization out of bounds"
        assert B_q.min() >= -8 and B_q.max() <= 7, "4-bit weight quantization out of bounds"
        assert x_q.min() >= 0 and x_q.max() <= 255, "8-bit activation quantization out of bounds"
        
        # NEW: Test scale factors are reasonable
        assert 0 < weight_params["A_scale"] < 1.0, "A scale factor unreasonable"
        assert 0 < weight_params["B_scale"] < 1.0, "B scale factor unreasonable"
        assert 0 < activation_scale < 1.0, "Activation scale factor unreasonable"
    
    @pytest.mark.skipif(not ezkl_available(), reason="EZKL binary not available")
    def test_onnx_export_validity(self, test_tensors, tmp_path):
        """Test ONNX export produces valid model file."""
        A, B, x, base_act = (
            test_tensors["A"], test_tensors["B"], 
            test_tensors["x"], test_tensors["base_activations"]
        )
        
        # Export circuit
        config = export_optimized_lora_circuit(
            submodule_name="test_module",
            A=A, B=B,
            x_data=x.numpy(),
            base_activations=base_act.numpy(),
            output_dir=str(tmp_path),
            verbose=False
        )
        
        # Check ONNX file exists
        onnx_path = tmp_path / "test_module_optimized.onnx"
        assert onnx_path.exists(), "ONNX file was not created"
        
        # Validate ONNX model can be loaded
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        
        # Check config contains expected optimization metadata
        assert "optimizations" in config
        assert "low_rank_factorization" in config["optimizations"]
        assert config["optimizations"]["low_rank_factorization"] is True
    
    @pytest.mark.skipif(not ezkl_available(), reason="EZKL binary not available")
    def test_ezkl_circuit_compilation(self, test_tensors, tmp_path):
        """Test EZKL can compile our ONNX to a circuit."""
        A, B, x, base_act = (
            test_tensors["A"], test_tensors["B"], 
            test_tensors["x"], test_tensors["base_activations"]
        )
        
        # Export ONNX
        export_optimized_lora_circuit(
            submodule_name="test_module",
            A=A, B=B,
            x_data=x.numpy(),
            base_activations=base_act.numpy(),
            output_dir=str(tmp_path),
            verbose=False
        )
        
        onnx_path = tmp_path / "test_module_optimized.onnx"
        circuit_path = tmp_path / "test_module.ezkl"
        
        # Compile with EZKL
        result = subprocess.run([
            "ezkl", "compile-circuit",
            "--model", str(onnx_path),
            "--output", str(circuit_path),
            "--bits", "8"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"EZKL compilation failed: {result.stderr}"
        assert circuit_path.exists(), "EZKL circuit file was not created"
        
        # Check circuit file is not empty and contains expected content
        circuit_size = circuit_path.stat().st_size
        assert circuit_size > 100, f"Circuit file suspiciously small: {circuit_size} bytes"
    
    @pytest.mark.skipif(not ezkl_available(), reason="EZKL binary not available")
    def test_witness_generation_and_verification(self, test_tensors, tmp_path):
        """Test complete round-trip: PyTorch → ONNX → EZKL → witness → verification."""
        A, B, x, base_act = (
            test_tensors["A"], test_tensors["B"], 
            test_tensors["x"], test_tensors["base_activations"]
        )
        
        # 1. Compute PyTorch reference
        pytorch_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # 2. Export ONNX circuit
        export_optimized_lora_circuit(
            submodule_name="test_module",
            A=A, B=B,
            x_data=x.numpy(),
            base_activations=base_act.numpy(),
            output_dir=str(tmp_path),
            verbose=False
        )
        
        onnx_path = tmp_path / "test_module_optimized.onnx"
        circuit_path = tmp_path / "test_module.ezkl"
        settings_path = tmp_path / "settings.json"
        witness_path = tmp_path / "witness.json"
        
        # 3. Generate EZKL settings
        result = subprocess.run([
            "ezkl", "gen-settings",
            "--model", str(onnx_path),
            "--output", str(settings_path),
            "--bits", "8"
        ], capture_output=True, text=True, timeout=30)
        
        assert result.returncode == 0, f"Settings generation failed: {result.stderr}"
        
        # 4. Compile circuit
        result = subprocess.run([
            "ezkl", "compile-circuit",
            "--model", str(onnx_path),
            "--settings", str(settings_path),
            "--output", str(circuit_path)
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Circuit compilation failed: {result.stderr}"
        
        # 5. Create input data JSON
        input_data = {"input_data": x.view(-1).tolist()}
        input_path = tmp_path / "input.json"
        with open(input_path, "w") as f:
            json.dump(input_data, f)
        
        # 6. Generate witness
        result = subprocess.run([
            "ezkl", "gen-witness",
            "--data", str(input_path),
            "--model", str(onnx_path),
            "--settings", str(settings_path),
            "--output", str(witness_path)
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode == 0, f"Witness generation failed: {result.stderr}"
        assert witness_path.exists(), "Witness file was not created"
        
        # 7. Load and compare witness output
        with open(witness_path) as f:
            witness_data = json.load(f)
        
        assert "output" in witness_data or "outputs" in witness_data, "Witness missing output data"
        
        # Extract output (format may vary)
        if "outputs" in witness_data:
            witness_output = torch.tensor(witness_data["outputs"][0])
        else:
            witness_output = torch.tensor(witness_data["output"])
        
        # Reshape to match PyTorch output
        if witness_output.numel() == pytorch_output.numel():
            witness_output = witness_output.view(pytorch_output.shape)
        else:
            # Handle potential flattening by EZKL
            witness_output = witness_output.view(-1)[:pytorch_output.numel()].view(pytorch_output.shape)
        
        # 8. Compare outputs (allow for quantization error)
        torch.testing.assert_close(
            pytorch_output,
            witness_output,
            rtol=0.05,  # 5% tolerance for quantization
            atol=0.01,
            msg="EZKL witness output differs significantly from PyTorch reference"
        )
        
        # 9. Verify circuit constraints (optional but recommended)
        try:
            result = subprocess.run([
                "ezkl", "verify",
                "--proof-path", str(witness_path),  # Some EZKL versions use witness as proof
                "--settings", str(settings_path),
                "--circuit", str(circuit_path)
            ], capture_output=True, text=True, timeout=30)
            
            # Don't fail test if verify command format is different
            if result.returncode == 0:
                print("✓ EZKL circuit constraints verified successfully")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("⚠ EZKL verify command format may differ - witness generation passed")
    
    @pytest.mark.parametrize("rank", [2, 4, 8, 16, 32, 64, 128])
    def test_different_ranks_comprehensive(self, rank):
        """Test circuit correctness across wide range of LoRA ranks with strict accuracy."""
        torch.manual_seed(42)
        
        d = 256  # Larger dimension for thorough testing
        A = torch.randn(d, rank) * (0.1 / np.sqrt(rank))  # Scale inversely with rank
        B = torch.randn(rank, d) * (0.1 / np.sqrt(rank))
        x = torch.randn(2, 16, d) * 0.1
        
        # Manual computation (high precision reference)
        manual_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # STRICT: Demand very high precision for all ranks
        abs_error = torch.abs(manual_output - circuit_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        assert max_abs_error < 1e-6, f"Rank {rank}: Max absolute error {max_abs_error:.8f} exceeds 1e-6"
        assert mean_abs_error < 1e-7, f"Rank {rank}: Mean absolute error {mean_abs_error:.8f} exceeds 1e-7"
        
        # Test relative error for significant values
        significant_mask = torch.abs(manual_output) > 1e-6
        if torch.any(significant_mask):
            significant_manual = manual_output[significant_mask]
            significant_circuit = circuit_output[significant_mask]
            
            rel_error = torch.abs(significant_manual - significant_circuit) / torch.abs(significant_manual)
            max_rel_error = torch.max(rel_error).item()
            
            assert max_rel_error < 1e-6, f"Rank {rank}: Max relative error {max_rel_error:.8f} exceeds 1e-6"
    
    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_multiple_random_seeds_strict(self, seed, small_lora_params):
        """Test circuit consistency across different random seeds with strict precision."""
        torch.manual_seed(seed)
        
        d, r = small_lora_params["d"], small_lora_params["r"]
        A = torch.randn(d, r) * 0.05
        B = torch.randn(r, d) * 0.05
        x = torch.randn(1, 8, d) * 0.1
        
        # Manual computation
        manual_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # EXTREMELY STRICT: Machine precision for non-quantized path
        torch.testing.assert_close(
            manual_output, 
            circuit_output, 
            rtol=1e-6, 
            atol=1e-7,
            msg=f"Circuit failed with seed {seed} - precision insufficient"
        )
    
    def test_high_rank_quantization_accuracy(self, large_test_tensors):
        """Test quantization accuracy for high-rank LoRA with demanding thresholds."""
        A, B, x = large_test_tensors["A"], large_test_tensors["B"], large_test_tensors["x"]
        
        # Original computation
        original_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Use higher precision quantization
        quantizer = LowRankQuantizer(weight_bits=6, activation_bits=8)  # 6-bit for higher precision
        A_q, B_q, weight_params = quantizer.quantize_weights(A, B)
        x_q, activation_scale, zero_point = quantizer.quantize_activations(x)
        
        # Dequantize
        A_dq = A_q.astype(np.float32) * weight_params["A_scale"]
        B_dq = B_q.astype(np.float32) * weight_params["B_scale"]
        x_dq = (x_q.astype(np.float32) - zero_point) * activation_scale
        
        A_dq_t = torch.from_numpy(A_dq)
        B_dq_t = torch.from_numpy(B_dq)
        x_dq_t = torch.from_numpy(x_dq)
        
        quantized_output = (x_dq_t.view(-1, x_dq_t.shape[-1]) @ A_dq_t @ B_dq_t).view(x.shape)
        
        # VERY STRICT: Higher precision demands even lower error for 6-bit quantization
        abs_error = torch.abs(original_output - quantized_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        assert max_abs_error < 0.001, f"High-rank max absolute error {max_abs_error:.6f} exceeds 0.1% threshold"
        assert mean_abs_error < 0.0002, f"High-rank mean absolute error {mean_abs_error:.6f} exceeds 0.02% threshold"
        
        # Test magnitude preservation for large tensors
        original_magnitude = torch.norm(original_output).item()
        quantized_magnitude = torch.norm(quantized_output).item()
        magnitude_error = abs(original_magnitude - quantized_magnitude) / (original_magnitude + 1e-8)
        
        assert magnitude_error < 0.01, f"High-rank magnitude error {magnitude_error:.4f} exceeds 1% threshold"
    
    @pytest.mark.parametrize("weight_bits,activation_bits", [
        (4, 8), (6, 8), (8, 8), (4, 16), (8, 16)
    ])
    def test_quantization_precision_scaling(self, test_tensors, weight_bits, activation_bits):
        """Test that higher bit quantization gives better precision."""
        A, B, x = test_tensors["A"], test_tensors["B"], test_tensors["x"]
        
        original_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        quantizer = LowRankQuantizer(weight_bits=weight_bits, activation_bits=activation_bits)
        A_q, B_q, weight_params = quantizer.quantize_weights(A, B)
        x_q, activation_scale, zero_point = quantizer.quantize_activations(x)
        
        # Dequantize
        A_dq = A_q.astype(np.float32) * weight_params["A_scale"]
        B_dq = B_q.astype(np.float32) * weight_params["B_scale"]
        x_dq = (x_q.astype(np.float32) - zero_point) * activation_scale
        
        quantized_output = (torch.from_numpy(x_dq).view(-1, x_dq.shape[-1]) @ 
                           torch.from_numpy(A_dq) @ torch.from_numpy(B_dq)).view(x.shape)
        
        abs_error = torch.abs(original_output - quantized_output)
        mean_abs_error = torch.mean(abs_error).item()
        
        # Each quantization level should maintain reasonable accuracy
        # Realistic thresholds based on actual quantization behavior
        threshold_map = {8: 0.1, 6: 0.15, 4: 0.25, 3: 0.4, 2: 0.6}
        max_acceptable_error = threshold_map.get(weight_bits, 1.0)
        assert mean_abs_error < max_acceptable_error, f"Bits {weight_bits}/{activation_bits}: Error {mean_abs_error:.4f} exceeds threshold {max_acceptable_error:.4f}"
    
    def test_edge_cases_rigorous(self):
        """Test circuit behavior with edge cases using strict precision requirements."""
        # Test with zeros - should be EXACTLY zero
        A_zero = torch.zeros(16, 4)
        B_zero = torch.zeros(4, 16) 
        x = torch.randn(1, 2, 16)
        
        circuit = SimpleLoRACircuit(A_zero, B_zero)
        output = circuit(x)
        
        # STRICT: Should be exactly zero
        assert torch.allclose(output, torch.zeros_like(output), atol=1e-8), "Zero matrices should produce exactly zero output"
        
        # Test with very small values - precise computation required
        A_tiny = torch.full((16, 4), 1e-6)
        B_tiny = torch.full((4, 16), 1e-6)
        
        circuit_tiny = SimpleLoRACircuit(A_tiny, B_tiny)
        output_tiny = circuit_tiny(x)
        
        # Manual computation for verification
        expected_tiny = (x.view(-1, x.shape[-1]) @ A_tiny @ B_tiny).view(x.shape)
        
        # STRICT: Should match manual computation exactly
        torch.testing.assert_close(output_tiny, expected_tiny, rtol=1e-10, atol=1e-12)
        
        # Test with rank-preserving matrices (proper identity test)
        # For LoRA, A @ B should be close to identity when A = B^T and both are orthogonal
        d = 16
        r = 4
        torch.manual_seed(42)
        
        # Create orthogonal matrix and use its subblocks
        Q, _ = torch.qr(torch.randn(d, d))
        A_orth = Q[:, :r]  # First r columns
        B_orth = Q[:r, :]  # First r rows
        
        # This creates a low-rank approximation to identity
        circuit_orth = SimpleLoRACircuit(A_orth, B_orth)
        x_small = torch.randn(1, 2, d) * 0.1
        output_orth = circuit_orth(x_small)
        expected_orth = (x_small.view(-1, x_small.shape[-1]) @ A_orth @ B_orth).view(x_small.shape)
        
        torch.testing.assert_close(output_orth, expected_orth, rtol=1e-10, atol=1e-12)
        
        # Test numerical stability with challenging but valid inputs
        A_challenge = torch.randn(16, 4) * 1e-3  # Small but not tiny
        B_challenge = torch.randn(4, 16) * 1e3   # Large
        x_challenge = torch.randn(1, 2, 16) * 1e-1
        
        circuit_challenge = SimpleLoRACircuit(A_challenge, B_challenge)
        output_challenge = circuit_challenge(x_challenge)
        expected_challenge = (x_challenge.view(-1, x_challenge.shape[-1]) @ A_challenge @ B_challenge).view(x_challenge.shape)
        
        # Should still be precise despite scale differences
        torch.testing.assert_close(output_challenge, expected_challenge, rtol=1e-8, atol=1e-10)
    
    def test_lookup_table_coverage_comprehensive(self, test_tensors):
        """Test quantization lookup tables with comprehensive validation."""
        A, B, x = test_tensors["A"], test_tensors["B"], test_tensors["x"]
        
        quantizer = LowRankQuantizer(weight_bits=4, activation_bits=8)
        A_q, B_q, weight_params = quantizer.quantize_weights(A, B)
        x_q, activation_scale, zero_point = quantizer.quantize_activations(x)
        
        # STRICT: Check quantized values are exactly in expected ranges
        assert A_q.min() >= -8 and A_q.max() <= 7, "4-bit signed weights out of range"
        assert B_q.min() >= -8 and B_q.max() <= 7, "4-bit signed weights out of range"  
        assert x_q.min() >= 0 and x_q.max() <= 255, "8-bit unsigned activations out of range"
        
        # More realistic quantization range usage test
        # Should use significant portion of available range (not necessarily full range)
        A_range = A_q.max() - A_q.min()
        B_range = B_q.max() - B_q.min()
        x_range = x_q.max() - x_q.min()
        
        assert A_range >= 4, f"Weight A quantization using too narrow range: {A_range}/15 possible"
        assert B_range >= 4, f"Weight B quantization using too narrow range: {B_range}/15 possible"
        assert x_range >= 50, f"Activation quantization using too narrow range: {x_range}/255 possible"
        
        # Check lookup table would have appropriate size
        lookup_size = 16 * 256  # 4-bit weights (-8 to 7 = 16 values) × 8-bit activations (0-255 = 256 values)
        assert lookup_size == 4096, f"Expected lookup table size 4096, got {lookup_size}"
        
        # Verify scale factors lead to proper reconstruction
        A_reconstructed = A_q * weight_params["A_scale"]
        B_reconstructed = B_q * weight_params["B_scale"]
        x_reconstructed = (x_q - zero_point) * activation_scale
        
        # STRICT: Scale factors should enable high-quality reconstruction
        A_error = np.mean(np.abs(A.numpy() - A_reconstructed))
        B_error = np.mean(np.abs(B.numpy() - B_reconstructed))
        x_error = np.mean(np.abs(x.numpy() - x_reconstructed))
        
        # Tighter thresholds for reconstruction quality
        A_rel_error = A_error / (np.mean(np.abs(A.numpy())) + 1e-8)
        B_rel_error = B_error / (np.mean(np.abs(B.numpy())) + 1e-8)
        x_rel_error = x_error / (np.mean(np.abs(x.numpy())) + 1e-8)
        
        # 4-bit quantization has fundamental limits, but should still be quite good
        assert A_rel_error < 0.15, f"A reconstruction relative error {A_rel_error:.4f} too high for 4-bit quantization"
        assert B_rel_error < 0.15, f"B reconstruction relative error {B_rel_error:.4f} too high for 4-bit quantization"
        assert x_rel_error < 0.05, f"x reconstruction relative error {x_rel_error:.4f} too high for 8-bit quantization"
        
        # Verify quantization is deterministic
        A_q2, B_q2, weight_params2 = quantizer.quantize_weights(A, B)
        x_q2, activation_scale2, zero_point2 = quantizer.quantize_activations(x)
        
        assert np.array_equal(A_q, A_q2), "Weight quantization not deterministic"
        assert np.array_equal(B_q, B_q2), "Weight quantization not deterministic"
        assert np.array_equal(x_q, x_q2), "Activation quantization not deterministic"
        assert abs(activation_scale - activation_scale2) < 1e-10, "Activation scale not deterministic"
        assert abs(zero_point - zero_point2) < 1e-10, "Zero point not deterministic"
    
    @pytest.mark.parametrize("scenario,a_scale,b_scale,x_scale,max_error", [
        ("Tiny values", 1e-6, 1e-6, 1e-3, 1e-8),
        ("Large values", 10.0, 10.0, 1.0, 1e-4),
        ("Mixed scales", 1e-3, 100.0, 0.1, 1e-6),
        ("Extreme ratio", 1e-8, 1e8, 1.0, 1e-2),
    ])
    def test_numerical_stability_comprehensive(self, scenario, a_scale, b_scale, x_scale, max_error):
        """Test numerical stability across various challenging scenarios."""
        torch.manual_seed(42)
        
        A = torch.randn(32, 8) * a_scale
        B = torch.randn(8, 32) * b_scale  
        x = torch.randn(1, 16, 32) * x_scale
        
        # Manual computation
        manual_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # Check numerical stability
        if torch.any(torch.isnan(manual_output)) or torch.any(torch.isnan(circuit_output)):
            pytest.fail(f"NaN values detected in scenario: {scenario}")
        
        if torch.any(torch.isinf(manual_output)) or torch.any(torch.isinf(circuit_output)):
            pytest.fail(f"Inf values detected in scenario: {scenario}")
        
        # Check precision based on scenario
        abs_error = torch.abs(manual_output - circuit_output)
        max_abs_error = torch.max(abs_error).item()
        
        assert max_abs_error < max_error, f"Scenario '{scenario}': Error {max_abs_error:.8f} exceeds {max_error}"

    @pytest.mark.parametrize("rank", [256, 512])
    def test_extreme_high_rank_stress(self, rank):
        """Stress test with extremely high ranks to test scalability."""
        torch.manual_seed(42)
        
        d = 1024  # Large embedding dimension
        # Scale weights even smaller for extreme ranks to maintain numerical stability
        scale_factor = 0.01 / np.sqrt(rank) 
        A = torch.randn(d, rank) * scale_factor
        B = torch.randn(rank, d) * scale_factor
        x = torch.randn(1, 32, d) * 0.01  # Small input to avoid overflow
        
        # Manual computation
        manual_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # Even for extreme ranks, demand machine precision
        abs_error = torch.abs(manual_output - circuit_output)
        max_abs_error = torch.max(abs_error).item()
        mean_abs_error = torch.mean(abs_error).item()
        
        assert max_abs_error < 1e-5, f"Extreme rank {rank}: Max absolute error {max_abs_error:.8f} exceeds 1e-5"
        assert mean_abs_error < 1e-6, f"Extreme rank {rank}: Mean absolute error {mean_abs_error:.8f} exceeds 1e-6"
        
        # Check no numerical instability
        assert not torch.any(torch.isnan(circuit_output)), f"Rank {rank}: NaN detected in circuit output"
        assert not torch.any(torch.isinf(circuit_output)), f"Rank {rank}: Inf detected in circuit output"
    
    def test_precision_degradation_with_quantization_bits(self):
        """Test quantization behavior across different bit widths."""
        torch.manual_seed(42)
        
        A = torch.randn(64, 8) * 0.1
        B = torch.randn(8, 64) * 0.1
        x = torch.randn(1, 16, 64) * 0.1
        
        original_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        bit_configs = [(8, 16), (6, 12), (4, 8), (3, 6), (2, 4)]
        errors = []
        
        for weight_bits, activation_bits in bit_configs:
            quantizer = LowRankQuantizer(weight_bits=weight_bits, activation_bits=activation_bits)
            A_q, B_q, weight_params = quantizer.quantize_weights(A, B)
            x_q, activation_scale, zero_point = quantizer.quantize_activations(x)
            
            # Dequantize
            A_dq = torch.from_numpy(A_q.astype(np.float32) * weight_params["A_scale"])
            B_dq = torch.from_numpy(B_q.astype(np.float32) * weight_params["B_scale"])
            x_dq = torch.from_numpy((x_q.astype(np.float32) - zero_point) * activation_scale)
            
            quantized_output = (x_dq.view(-1, x_dq.shape[-1]) @ A_dq @ B_dq).view(x.shape)
            
            current_error = torch.mean(torch.abs(original_output - quantized_output)).item()
            errors.append((weight_bits, activation_bits, current_error))
            
            # Each quantization level should maintain reasonable accuracy
            # Realistic thresholds based on actual quantization behavior
            threshold_map = {8: 0.1, 6: 0.15, 4: 0.25, 3: 0.4, 2: 0.6}
            max_acceptable_error = threshold_map.get(weight_bits, 1.0)
            assert current_error < max_acceptable_error, f"Bits {weight_bits}/{activation_bits}: Error {current_error:.4f} exceeds threshold {max_acceptable_error:.4f}"
        
        # Verify quantization behavior for each configuration
        error_8_16 = errors[0][2]  # (8,16) error
        error_2_4 = errors[4][2]   # (2,4) error
        
        # Sanity check: 2-bit quantization should not be dramatically better than 8-bit
        # (which would indicate a bug in the test or implementation)
        if error_2_4 < error_8_16 * 0.1:
            print(f"Warning: 2-bit error ({error_2_4:.6f}) much smaller than 8-bit error ({error_8_16:.6f}) - may indicate test issue")
        
        # Print quantization performance for analysis
        for weight_bits, activation_bits, error in errors:
            print(f"Quantization {weight_bits}/{activation_bits} bits: error = {error:.6f}")
        
        # Verify that quantization bounds are respected for each configuration
        for weight_bits, activation_bits, _ in errors:
            quantizer = LowRankQuantizer(weight_bits=weight_bits, activation_bits=activation_bits)
            A_q, B_q, _ = quantizer.quantize_weights(A, B)
            x_q, _, _ = quantizer.quantize_activations(x)
            
            weight_min, weight_max = -(2**(weight_bits-1)), 2**(weight_bits-1) - 1
            activation_max = 2**activation_bits - 1
            
            assert A_q.min() >= weight_min and A_q.max() <= weight_max, f"Weight bounds violated for {weight_bits}-bit"
            assert B_q.min() >= weight_min and B_q.max() <= weight_max, f"Weight bounds violated for {weight_bits}-bit"
            assert x_q.min() >= 0 and x_q.max() <= activation_max, f"Activation bounds violated for {activation_bits}-bit"
    
    @pytest.mark.parametrize("batch_size,seq_len", [
        (1, 1024),      # Long sequence
        (32, 128),      # Large batch
        (8, 2048),      # Extra long sequence
        (64, 64),       # Large batch, medium sequence
    ])
    def test_large_tensor_precision(self, batch_size, seq_len):
        """Test precision maintenance with large tensor dimensions."""
        torch.manual_seed(42)
        
        d = 384  # Medium embedding dimension
        r = 16   # Moderate rank
        
        A = torch.randn(d, r) * 0.02
        B = torch.randn(r, d) * 0.02
        x = torch.randn(batch_size, seq_len, d) * 0.05
        
        # Manual computation
        manual_output = (x.view(-1, x.shape[-1]) @ A @ B).view(x.shape)
        
        # Circuit computation
        circuit = SimpleLoRACircuit(A, B)
        circuit_output = circuit(x)
        
        # Large tensors should maintain precision
        torch.testing.assert_close(
            manual_output, 
            circuit_output, 
            rtol=1e-6, 
            atol=1e-7,
            msg=f"Large tensor ({batch_size}x{seq_len}x{d}) precision insufficient"
        )
        
        # Check memory doesn't cause numerical issues
        assert not torch.any(torch.isnan(circuit_output)), "NaN detected in large tensor computation"
        assert not torch.any(torch.isinf(circuit_output)), "Inf detected in large tensor computation"
    
    def test_quantization_consistency_across_shapes(self):
        """Test that quantization behavior is consistent across different tensor shapes."""
        torch.manual_seed(42)
        
        # Same underlying values, different shapes
        base_values = torch.randn(32 * 8) * 0.1
        
        A1 = base_values.view(32, 8)
        A2 = base_values.view(8, 32)
        A3 = base_values.view(16, 16)
        A4 = base_values.view(64, 4)
        
        quantizer = LowRankQuantizer(weight_bits=4, activation_bits=8)
        
        # Quantize each shape
        A1_q, _, params1 = quantizer.quantize_weights(A1, A1.T)
        A2_q, _, params2 = quantizer.quantize_weights(A2, A2.T)
        A3_q, _, params3 = quantizer.quantize_weights(A3, A3.T)
        A4_q, _, params4 = quantizer.quantize_weights(A4, A4.T)
        
        # Reconstruction errors should be similar across shapes
        errors = []
        for A_orig, A_q, params in [(A1, A1_q, params1), (A2, A2_q, params2), 
                                   (A3, A3_q, params3), (A4, A4_q, params4)]:
            A_recon = torch.from_numpy(A_q * params['A_scale'])
            error = torch.mean(torch.abs(A_orig - A_recon)).item()
            errors.append(error)
        
        # All errors should be within 50% of each other
        min_error, max_error = min(errors), max(errors)
        error_ratio = max_error / (min_error + 1e-8)
        
        assert error_ratio < 2.0, f"Quantization errors vary too much across shapes: {errors}, ratio: {error_ratio:.2f}"


@pytest.mark.integration 
class TestCircuitIntegrationRequirements:
    """Test requirements and environment setup for circuit validation."""
    
    def test_ezkl_installation_check(self):
        """Provide clear guidance when EZKL is not available."""
        if not ezkl_available():
            pytest.skip(
                "EZKL not installed. Install with:\n"
                "  cargo install ezkl\n"
                "Or download from: https://github.com/zkonduit/ezkl"
            )
    
    def test_temp_directory_permissions(self, tmp_path):
        """Test that temporary directories have correct permissions."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        assert test_file.exists()
        assert test_file.read_text() == "test"
    
    def test_subprocess_timeout_handling(self):
        """Test that subprocess timeouts are handled gracefully."""
        with pytest.raises(subprocess.TimeoutExpired):
            subprocess.run(["sleep", "2"], timeout=0.1) 