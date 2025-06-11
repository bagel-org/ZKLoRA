import pytest
import torch
import numpy as np
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import torch.nn as nn # For creating mock submodules
import os # For os.path.join if needed in asserts

# Adjust sys.path to include the 'src' directory, parent of 'zklora' package
_P = Path
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "src"))

from zklora.mpi_lora_onnx_exporter import normalize_lora_matrices_mpi, LoraShapeTransformerMPI

class TestNormalizeLoraMatricesMPI:
    def test_correct_shapes_no_transpose(self):
        A = torch.randn(10, 5)  # in_dim=10, r=5
        B = torch.randn(5, 8)   # r=5, out_dim=8
        x_data = np.random.randn(1, 1, 10) # batch, seq, hidden_dim(in_dim)
        A_fixed, B_fixed, in_dim, r, out_dim = normalize_lora_matrices_mpi(A, B, x_data)
        assert torch.equal(A_fixed, A)
        assert torch.equal(B_fixed, B)
        assert in_dim == 10
        assert r == 5
        assert out_dim == 8

    def test_A_needs_transpose(self):
        A_orig = torch.randn(5, 10) # r=5, in_dim=10 (transposed)
        B = torch.randn(5, 8)      # r=5, out_dim=8
        x_data = np.random.randn(1, 1, 10)
        A_fixed, B_fixed, in_dim, r, out_dim = normalize_lora_matrices_mpi(A_orig, B, x_data)
        assert torch.equal(A_fixed, A_orig.transpose(0, 1))
        assert torch.equal(B_fixed, B)
        assert in_dim == 10
        assert r == 5
        assert out_dim == 8

    def test_B_needs_transpose(self):
        A = torch.randn(10, 5)      # in_dim=10, r=5
        B_orig = torch.randn(8, 5)  # out_dim=8, r=5 (transposed)
        x_data = np.random.randn(1, 1, 10)
        A_fixed, B_fixed, in_dim, r, out_dim = normalize_lora_matrices_mpi(A, B_orig, x_data)
        assert torch.equal(A_fixed, A)
        assert torch.equal(B_fixed, B_orig.transpose(0, 1))
        assert in_dim == 10
        assert r == 5
        assert out_dim == 8

    def test_A_and_B_need_transpose(self):
        A_orig = torch.randn(5, 10) # r=5, in_dim=10 (transposed)
        B_orig = torch.randn(8, 5)  # out_dim=8, r=5 (transposed)
        x_data = np.random.randn(1, 1, 10)
        A_fixed, B_fixed, in_dim, r, out_dim = normalize_lora_matrices_mpi(A_orig, B_orig, x_data)
        assert torch.equal(A_fixed, A_orig.transpose(0, 1))
        assert torch.equal(B_fixed, B_orig.transpose(0, 1))
        assert in_dim == 10
        assert r == 5
        assert out_dim == 8

    def test_A_shape_mismatch_error(self):
        A = torch.randn(12, 5) # in_dim=12, but x_data has 10
        B = torch.randn(5, 8)
        x_data = np.random.randn(1, 1, 10)
        with pytest.raises(ValueError, match=r"A shape .* doesn't match x_data last dim 10"):
            normalize_lora_matrices_mpi(A, B, x_data)

    def test_B_shape_mismatch_error(self):
        A = torch.randn(10, 5) # in_dim=10, r=5
        B = torch.randn(6, 8)  # r should be 5, but b0 is 6
        x_data = np.random.randn(1, 1, 10)
        with pytest.raises(ValueError, match=r"B shape .* doesn't match rank=5"):
            normalize_lora_matrices_mpi(A, B, x_data)

class TestLoraShapeTransformerMPI:
    def test_forward_pass_shape_and_values(self):
        A_val = torch.tensor([[1., 2.], [3., 4.]])  # hidden_dim=2, r=2
        B_val = torch.tensor([[0.5, 1.5], [2.5, 3.5]]) # r=2, out_dim=2 (same as hidden_dim for this test)
        batch_size = 1
        seq_len = 3
        hidden_dim = 2 # Must match A's input dim and B's output dim for this calculation

        transformer = LoraShapeTransformerMPI(A_val, B_val, batch_size, seq_len, hidden_dim)

        # x_1d shape: (1, batch_size * seq_len * hidden_dim)
        x_input_1d = torch.arange(1, batch_size * seq_len * hidden_dim + 1, dtype=torch.float32).view(1, -1)
        # x_input_1d will be [[1, 2, 3, 4, 5, 6]] for (1,3,2)
        # x_3d will be [[[1,2], [3,4], [5,6]]]

        output_2d = transformer(x_input_1d)

        # Expected output shape (1, batch_size * seq_len * out_dim)
        # Since out_dim (from B) is hidden_dim for this test, it's (1, 1*3*2) = (1,6)
        assert output_2d.shape == (1, batch_size * seq_len * hidden_dim)

        # Calculate expected values manually
        x_3d_manual = x_input_1d.view(batch_size, seq_len, hidden_dim)
        # (x_3d @ A) @ B
        lora_out_manual = (x_3d_manual @ A_val) @ B_val
        # out_3d = out_3d + x_3d.mean() + self.A.sum() + self.B.sum()
        expected_out_3d = lora_out_manual + x_3d_manual.mean() + A_val.sum() + B_val.sum()
        expected_out_2d_manual = expected_out_3d.view(1, -1)

        assert torch.allclose(output_2d, expected_out_2d_manual)

    def test_different_batch_seq_dims(self):
        A_val = torch.randn(4, 2) # hidden_dim=4, r=2
        B_val = torch.randn(2, 3) # r=2, out_dim=3
        batch_size = 2
        seq_len = 5
        hidden_dim = 4

        transformer = LoraShapeTransformerMPI(A_val, B_val, batch_size, seq_len, hidden_dim)
        x_input_1d = torch.randn(1, batch_size * seq_len * hidden_dim)
        output_2d = transformer(x_input_1d)

        # Output shape should be (1, batch_size * seq_len * out_dim from B)
        # But the current LoraShapeTransformerMPI always reshapes to hidden_dim in output.
        # The problem description out_3d = (x_3d @ self.A) @ self.B implicitly defines output hidden_dim
        # So, the output will be (1, batch_size * seq_len * hidden_dim_of_B_output)
        # However, the LoraShapeTransformerMPI current code uses self.hidden_dim for the output view calculation.
        # This test will follow the current code logic where out_dim of the transformer is effectively hidden_dim.
        # If B_val.shape[1] was to be used for true out_dim, the transformer code would need change.
        
        # The current code: out_3d.view(1,-1) and x_1d.view(self.batch_size, self.seq_len, self.hidden_dim)
        # The output of (x_3d @ self.A) @ self.B will have shape (batch_size, seq_len, B_val.shape[1])
        # So the output.view(1,-1) will be (1, batch_size * seq_len * B_val.shape[1])
        assert output_2d.shape == (1, batch_size * seq_len * B_val.shape[1]) 

class MockLoraLinearLayer(nn.Module):
    def __init__(self, weight_data):
        super().__init__()
        self.weight = nn.Parameter(weight_data)

class MockSubmodule(nn.Module):
    def __init__(self, has_lora_A=True, has_lora_B=True, a_keys=['default'], a_data=None, b_data=None):
        super().__init__()
        if has_lora_A:
            self.lora_A = nn.ModuleDict()
            if a_keys and a_data is not None:
                for key in a_keys:
                    self.lora_A[key] = MockLoraLinearLayer(a_data)
        if has_lora_B:
            self.lora_B = nn.ModuleDict()
            if a_keys and b_data is not None: # Assuming b_keys are same as a_keys
                for key in a_keys:
                    self.lora_B[key] = MockLoraLinearLayer(b_data)

# Need to import the function we are testing
from zklora.mpi_lora_onnx_exporter import export_lora_onnx_json_mpi

class TestExportLoraOnnxJsonMPI:
    @patch('zklora.mpi_lora_onnx_exporter.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.onnx.export')
    @patch('zklora.mpi_lora_onnx_exporter.normalize_lora_matrices_mpi')
    def test_successful_export(self, mock_normalize, mock_torch_export, mock_file_open, mock_makedirs, tmp_path):
        sub_name = "test.submodule"
        x_data = np.random.randn(1, 3, 10) # batch, seq, hidden
        output_dir = str(tmp_path)

        A_data = torch.randn(10, 5) # hidden, rank
        B_data = torch.randn(5, 10) # rank, hidden_out (same as hidden for this test)
        mock_sub = MockSubmodule(a_data=A_data, b_data=B_data)

        mock_normalize.return_value = (A_data, B_data, 10, 5, 10)

        export_lora_onnx_json_mpi(sub_name, x_data, mock_sub, output_dir, verbose=False)

        mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)
        mock_normalize.assert_called_once()
        mock_torch_export.assert_called_once()
        
        expected_safe_name = sub_name.replace(".", "_").replace("/", "_")
        expected_onnx_path = os.path.join(output_dir, f"{expected_safe_name}.onnx")
        args, kwargs = mock_torch_export.call_args
        assert args[2] == expected_onnx_path
        assert isinstance(args[0], LoraShapeTransformerMPI)

        expected_json_path = os.path.join(output_dir, f"{expected_safe_name}.json")
        mock_file_open.assert_called_once_with(expected_json_path, "w")

    @patch('zklora.mpi_lora_onnx_exporter.print') # To check verbose output
    @patch('zklora.mpi_lora_onnx_exporter.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.onnx.export')
    @patch('zklora.mpi_lora_onnx_exporter.normalize_lora_matrices_mpi')
    def test_skip_no_lora_A(self, mock_normalize, mock_torch_export, mock_file_open, mock_makedirs, mock_print, tmp_path):
        mock_sub = MockSubmodule(has_lora_A=False)
        export_lora_onnx_json_mpi("no_lora_A", np.random.randn(1,2,4), mock_sub, str(tmp_path), verbose=True)
        mock_normalize.assert_not_called()
        mock_torch_export.assert_not_called()
        mock_file_open.assert_not_called()
        mock_print.assert_any_call("[export_lora_onnx_json_mpi] No lora_A/B in submodule 'no_lora_A', skipping.")

    @patch('zklora.mpi_lora_onnx_exporter.print')
    @patch('zklora.mpi_lora_onnx_exporter.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.onnx.export')
    @patch('zklora.mpi_lora_onnx_exporter.normalize_lora_matrices_mpi')
    def test_skip_no_adapter_keys(self, mock_normalize, mock_torch_export, mock_file_open, mock_makedirs, mock_print, tmp_path):
        mock_sub = MockSubmodule(a_keys=[]) # No adapter keys
        export_lora_onnx_json_mpi("no_keys", np.random.randn(1,2,4), mock_sub, str(tmp_path), verbose=True)
        mock_normalize.assert_not_called()
        mock_torch_export.assert_not_called()
        mock_print.assert_any_call("[export_lora_onnx_json_mpi] No adapter keys in submodule.lora_A for 'no_keys'.")

    @patch('zklora.mpi_lora_onnx_exporter.print')
    @patch('zklora.mpi_lora_onnx_exporter.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.onnx.export')
    @patch('zklora.mpi_lora_onnx_exporter.normalize_lora_matrices_mpi', side_effect=ValueError("Shape error"))
    def test_skip_normalize_value_error(self, mock_normalize_error, mock_torch_export, mock_file_open, mock_makedirs, mock_print, tmp_path):
        A_data = torch.randn(10, 5)
        B_data = torch.randn(5, 10)
        mock_sub = MockSubmodule(a_data=A_data, b_data=B_data)
        export_lora_onnx_json_mpi("norm_error", np.random.randn(1,2,10), mock_sub, str(tmp_path), verbose=True)
        mock_normalize_error.assert_called_once()
        mock_torch_export.assert_not_called()
        mock_print.assert_any_call("Shape fix error for 'norm_error': Shape error")

    @patch('zklora.mpi_lora_onnx_exporter.print')
    @patch('zklora.mpi_lora_onnx_exporter.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('torch.onnx.export', side_effect=Exception("ONNX Export Failed"))
    @patch('zklora.mpi_lora_onnx_exporter.normalize_lora_matrices_mpi')
    def test_onnx_export_exception(self, mock_normalize, mock_torch_export_error, mock_file_open, mock_makedirs, mock_print, tmp_path):
        A_data = torch.randn(10, 5)
        B_data = torch.randn(5, 10)
        mock_sub = MockSubmodule(a_data=A_data, b_data=B_data)
        mock_normalize.return_value = (A_data, B_data, 10, 5, 10)
        
        export_lora_onnx_json_mpi("export_fail", np.random.randn(1,2,10), mock_sub, str(tmp_path), verbose=True)
        
        mock_normalize.assert_called_once()
        mock_torch_export_error.assert_called_once()
        # JSON should still be saved even if ONNX export fails, as per current code structure
        mock_file_open.assert_called_once() 
        mock_print.assert_any_call("Export error for 'export_fail': ONNX Export Failed")
        # Also check for the successful print messages if verbose is True