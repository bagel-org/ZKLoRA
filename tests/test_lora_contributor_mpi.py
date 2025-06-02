from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np
import threading
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock

from zklora.lora_contributor_mpi import (
    strip_prefix,
    read_file_as_bytes,
    LoRAServer,
    LoRAServerSocket
)


@pytest.mark.unit
def test_strip_prefix():
    # Test with base_model.model prefix
    assert strip_prefix("base_model.model.transformer.h.0.attn.c_attn") == "transformer.h.0.attn.c_attn"
    
    # Test with base_model prefix
    assert strip_prefix("base_model.transformer.h.0.attn.c_attn") == "transformer.h.0.attn.c_attn"
    
    # Test with model prefix
    assert strip_prefix("model.transformer.h.0.attn.c_attn") == "transformer.h.0.attn.c_attn"
    
    # Test with no prefix
    assert strip_prefix("transformer.h.0.attn.c_attn") == "transformer.h.0.attn.c_attn"
    
    # Test empty string
    assert strip_prefix("") == ""
    
    # Test string with extra spaces - strip() removes leading/trailing whitespace
    assert strip_prefix("  transformer.test  ") == "transformer.test"


@pytest.mark.unit
def test_read_file_as_bytes():
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        test_data = b"Hello, world!"
        tmp_file.write(test_data)
        tmp_file.flush()
        
        try:
            result = read_file_as_bytes(tmp_file.name)
            assert result == test_data
        finally:
            os.unlink(tmp_file.name)


@pytest.mark.unit
class TestLoRAServer:
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_initialization(self, mock_peft, mock_auto_model):
        # Mock base model
        mock_base = Mock()
        mock_base.config = Mock()
        mock_base.config.use_cache = True
        mock_auto_model.return_value = mock_base
        
        # Mock PEFT model with LoRA modules
        mock_peft_model = Mock()
        
        # Create mock LoRA module that passes the filters
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_param.name = "lora_A.weight"
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer(
                base_model_name="test_model",
                lora_model_id="test_lora",
                out_dir=tmpdir,
                use_optimization=True
            )
            
            assert server.out_dir == tmpdir
            assert server.use_optimization is True
            assert mock_base.config.use_cache is False
            assert mock_base.eval.called
            assert mock_peft_model.eval.called
            assert len(server.submodules) == 1
            assert "transformer.h.0.attn.c_attn" in server.submodules
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_initialization_filtering(self, mock_peft, mock_auto_model):
        # Test that modules are properly filtered
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        
        # Create various modules to test filtering
        mock_lora_module_good = Mock()
        mock_lora_param = Mock()
        mock_lora_module_good.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_no_lora_module = Mock()
        mock_no_lora_module.named_parameters.return_value = [("weight", Mock())]
        
        mock_lora_module_bad_name = Mock()
        mock_lora_module_bad_name.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module_good),  # Good
            ("base_model.model.transformer.h.0.mlp.c_fc", mock_no_lora_module),      # No LoRA
            ("base_model.model.transformer.h.0.attn.qkv", mock_lora_module_bad_name), # Bad ending
            ("", mock_lora_module_good),  # Empty name
            ("transformer", mock_lora_module_good),  # No dots
        ]
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir)
            
            # Only the good module should be included
            assert len(server.submodules) == 1
            assert "transformer.h.0.attn.c_attn" in server.submodules
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_list_lora_injection_points(self, mock_peft, mock_auto_model):
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module),
            ("base_model.model.transformer.h.1.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir)
            
            injection_points = server.list_lora_injection_points()
            assert len(injection_points) == 2
            assert "transformer.h.0.attn.c_attn" in injection_points
            assert "transformer.h.1.attn.c_attn" in injection_points
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_apply_lora_basic(self, mock_peft, mock_auto_model):
        # Setup mocks
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        # Mock the forward pass
        mock_output = torch.tensor([[1.0, 2.0, 3.0]])
        mock_lora_module.return_value = mock_output
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir)
            
            input_tensor = torch.tensor([[0.5, 1.0, 1.5]])
            result = server.apply_lora("transformer.h.0.attn.c_attn", input_tensor)
            
            assert torch.equal(result, mock_output)
            assert "transformer.h.0.attn.c_attn" in server.session_data
            assert len(server.session_data["transformer.h.0.attn.c_attn"]) == 1
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_apply_lora_with_base_activation(self, mock_peft, mock_auto_model):
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        mock_lora_module.return_value = torch.tensor([[1.0, 2.0, 3.0]])
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir)
            
            input_tensor = torch.tensor([[0.5, 1.0, 1.5]])
            base_activation = np.array([[0.1, 0.2, 0.3]])
            
            server.apply_lora("transformer.h.0.attn.c_attn", input_tensor, base_activation)
            
            assert "transformer.h.0.attn.c_attn" in server.base_activations
            assert len(server.base_activations["transformer.h.0.attn.c_attn"]) == 1
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_apply_lora_invalid_module(self, mock_peft, mock_auto_model):
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_peft_model.named_modules.return_value = []
        mock_peft.return_value = mock_peft_model
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir)
            
            input_tensor = torch.tensor([[0.5, 1.0, 1.5]])
            
            with pytest.raises(ValueError, match="not recognized"):
                server.apply_lora("nonexistent.module", input_tensor)
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    @patch('asyncio.run')
    @patch('zklora.lora_contributor_mpi.export_lora_onnx_json_mpi_optimized')
    def test_finalize_proofs_optimized(self, mock_export_opt, mock_asyncio, mock_peft, mock_auto_model):
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        mock_asyncio.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir, use_optimization=True)
            
            # Add some session data and base activations
            server.session_data["transformer.h.0.attn.c_attn"] = [np.array([[1, 2, 3]])]
            server.base_activations["transformer.h.0.attn.c_attn"] = [np.array([[0.1, 0.2, 0.3]])]
            
            server.finalize_proofs_and_collect()
            
            mock_export_opt.assert_called_once()
            mock_asyncio.assert_called_once()
            assert len(server.session_data) == 0  # Should be cleared
            assert len(server.base_activations) == 0  # Should be cleared
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    @patch('asyncio.run')
    @patch('zklora.lora_contributor_mpi.export_lora_onnx_json_mpi')
    def test_finalize_proofs_non_optimized(self, mock_export, mock_asyncio, mock_peft, mock_auto_model):
        mock_base = Mock()
        mock_base.config = Mock()
        mock_auto_model.return_value = mock_base
        
        mock_peft_model = Mock()
        mock_lora_module = Mock()
        mock_lora_param = Mock()
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model.named_modules.return_value = [
            ("base_model.model.transformer.h.0.attn.c_attn", mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        mock_asyncio.return_value = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            server = LoRAServer("test_model", "test_lora", tmpdir, use_optimization=False)
            
            # Add session data without base activations
            server.session_data["transformer.h.0.attn.c_attn"] = [np.array([[1, 2, 3]])]
            
            server.finalize_proofs_and_collect()
            
            mock_export.assert_called_once()
            mock_asyncio.assert_called_once()


@pytest.mark.unit 
class TestLoRAServerSocket:
    
    def test_initialization(self):
        server = Mock()
        stop_event = threading.Event()
        
        socket_server = LoRAServerSocket("127.0.0.1", 30000, server, stop_event)
        
        assert socket_server.host == "127.0.0.1"
        assert socket_server.port == 30000
        assert socket_server.lora_server == server
        assert socket_server.stop_event == stop_event
    
    def test_recv_all_basic(self):
        server = Mock()
        stop_event = threading.Event()
        socket_server = LoRAServerSocket("127.0.0.1", 30000, server, stop_event)
        
        # Mock connection
        mock_conn = Mock()
        mock_conn.recv.side_effect = [b"hello", b"world", b""]  # Empty bytes indicates end
        
        result = socket_server.recv_all(mock_conn)
        
        assert result == b"helloworld"
        mock_conn.settimeout.assert_called_with(1200.0)
    
    def test_recv_all_timeout(self):
        server = Mock()
        stop_event = threading.Event()
        socket_server = LoRAServerSocket("127.0.0.1", 30000, server, stop_event)
        
        # Mock connection with timeout
        mock_conn = Mock()
        import socket
        mock_conn.recv.side_effect = socket.timeout()
        
        result = socket_server.recv_all(mock_conn)
        
        assert result == b""
    
    @patch('socket.socket')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    def test_handle_conn_init_request(self, mock_dumps, mock_loads, mock_socket):
        mock_server = Mock()
        mock_server.list_lora_injection_points.return_value = ["test_module"]
        stop_event = threading.Event()
        
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        mock_loads.return_value = {"request_type": "init_request"}
        mock_dumps.return_value = b"response"
        
        # Mock recv_all
        socket_server.recv_all = Mock(return_value=b"request_data")
        
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        mock_server.list_lora_injection_points.assert_called_once()
        mock_conn.sendall.assert_called_once_with(b"response")
        mock_conn.close.assert_called_once()
    
    @patch('socket.socket')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    @patch('torch.tensor')
    def test_handle_conn_lora_forward(self, mock_tensor, mock_dumps, mock_loads, mock_socket):
        mock_server = Mock()
        mock_output = Mock()
        mock_output.cpu.return_value.numpy.return_value = [[1, 2, 3]]
        mock_server.apply_lora.return_value = mock_output
        
        stop_event = threading.Event()
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        mock_loads.return_value = {
            "request_type": "lora_forward",
            "submodule_name": "test_module",
            "input_array": [1, 2, 3]
        }
        mock_dumps.return_value = b"response"
        mock_tensor.return_value = torch.tensor([1, 2, 3])
        
        socket_server.recv_all = Mock(return_value=b"request_data")
        
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        mock_server.apply_lora.assert_called_once()
        mock_conn.sendall.assert_called_once_with(b"response")
    
    @patch('socket.socket')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    @patch('torch.tensor')
    def test_handle_conn_lora_forward_optimized(self, mock_tensor, mock_dumps, mock_loads, mock_socket):
        mock_server = Mock()
        mock_output = Mock()
        mock_output.cpu.return_value.numpy.return_value = [[1, 2, 3]]
        mock_server.apply_lora.return_value = mock_output
        
        stop_event = threading.Event()
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        mock_loads.return_value = {
            "request_type": "lora_forward_optimized",
            "submodule_name": "test_module",
            "input_array": [1, 2, 3],
            "base_activation": [0.1, 0.2, 0.3]
        }
        mock_dumps.return_value = b"response"
        mock_tensor.return_value = torch.tensor([1, 2, 3])
        
        socket_server.recv_all = Mock(return_value=b"request_data")
        
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        # Should call apply_lora with base activation
        args, kwargs = mock_server.apply_lora.call_args
        assert len(args) == 3  # submodule_name, tensor, base_activation
        mock_conn.sendall.assert_called_once_with(b"response")
    
    @patch('socket.socket')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    def test_handle_conn_end_inference(self, mock_dumps, mock_loads, mock_socket):
        mock_server = Mock()
        stop_event = threading.Event()
        
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        mock_loads.return_value = {"request_type": "end_inference"}
        mock_dumps.return_value = b"response"
        
        socket_server.recv_all = Mock(return_value=b"request_data")
        
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        mock_server.finalize_proofs_and_collect.assert_called_once()
        mock_conn.sendall.assert_called_once_with(b"response")
    
    @patch('socket.socket')
    @patch('pickle.loads')
    @patch('pickle.dumps')
    def test_handle_conn_unknown_request(self, mock_dumps, mock_loads, mock_socket):
        mock_server = Mock()
        stop_event = threading.Event()
        
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        mock_loads.return_value = {"request_type": "unknown_type"}
        mock_dumps.return_value = b"response"
        
        socket_server.recv_all = Mock(return_value=b"request_data")
        
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        mock_conn.sendall.assert_called_once_with(b"response")
    
    def test_handle_conn_exception(self):
        mock_server = Mock()
        stop_event = threading.Event()
        
        socket_server = LoRAServerSocket("127.0.0.1", 30000, mock_server, stop_event)
        
        mock_conn = Mock()
        socket_server.recv_all = Mock(side_effect=Exception("Test error"))
        
        # Should not raise exception, just handle it gracefully
        socket_server.handle_conn(mock_conn, ("127.0.0.1", 12345))
        
        mock_conn.close.assert_called_once() 