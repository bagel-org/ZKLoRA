from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from zklora.base_model_user_mpi import (
    BaseModelToLoRAComm,
    RemoteLoRAWrappedModule,
    BaseModelClient
)


@pytest.mark.unit
class TestBaseModelToLoRAComm:
    
    def test_initialization(self):
        comm = BaseModelToLoRAComm("192.168.1.1", 40000)
        assert comm.host_a == "192.168.1.1"
        assert comm.port_a == 40000
    
    def test_default_initialization(self):
        comm = BaseModelToLoRAComm()
        assert comm.host_a == "127.0.0.1"
        assert comm.port_a == 30000
    
    @patch('socket.socket')
    def test_init_request(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = [b'pickled_response', b'']
        
        with patch('pickle.loads') as mock_loads:
            mock_loads.return_value = {"injection_points": ["test_module"]}
            
            comm = BaseModelToLoRAComm()
            result = comm.init_request()
            
            assert result == ["test_module"]
            mock_sock.connect.assert_called_once_with(("127.0.0.1", 30000))
    
    @patch('socket.socket')
    def test_lora_forward(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = [b'pickled_response', b'']
        
        with patch('pickle.loads') as mock_loads, \
             patch('pickle.dumps') as mock_dumps:
            mock_loads.return_value = {"output_array": [1, 2, 3]}
            mock_dumps.return_value = b'pickled_request'
            
            comm = BaseModelToLoRAComm()
            result = comm.lora_forward("test_module", [1, 2, 3])
            
            assert result == [1, 2, 3]
    
    @patch('socket.socket')
    def test_lora_forward_with_base(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = [b'pickled_response', b'']
        
        with patch('pickle.loads') as mock_loads, \
             patch('pickle.dumps') as mock_dumps:
            mock_loads.return_value = {"output_array": [4, 5, 6]}
            mock_dumps.return_value = b'pickled_request'
            
            comm = BaseModelToLoRAComm()
            result = comm.lora_forward_with_base(
                "test_module", [1, 2, 3], [0.1, 0.2, 0.3]
            )
            
            assert result == [4, 5, 6]
    
    @patch('socket.socket')
    def test_end_inference(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = [b'pickled_response', b'']
        
        with patch('pickle.loads') as mock_loads:
            mock_loads.return_value = {"status": "complete"}
            
            comm = BaseModelToLoRAComm()
            result = comm.end_inference()
            
            assert result == {"status": "complete"}
    
    @patch('socket.socket')
    def test_send_and_recv_timeout(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        
        import socket
        mock_sock.recv.side_effect = socket.timeout()
        
        with patch('pickle.dumps') as mock_dumps:
            mock_dumps.return_value = b'pickled_request'
            
            comm = BaseModelToLoRAComm()
            
            with pytest.raises(RuntimeError, match="No data from A"):
                comm.send_and_recv({"test": "data"})
    
    @patch('socket.socket')
    def test_send_and_recv_success(self, mock_socket):
        mock_sock = Mock()
        mock_socket.return_value = mock_sock
        mock_sock.recv.side_effect = [b'pickled_response', b'']
        
        with patch('pickle.loads') as mock_loads, \
             patch('pickle.dumps') as mock_dumps:
            mock_loads.return_value = {"success": True}
            mock_dumps.return_value = b'pickled_request'
            
            comm = BaseModelToLoRAComm()
            result = comm.send_and_recv({"test": "data"})
            
            assert result == {"success": True}
            mock_sock.settimeout.assert_called_with(1200.0)


@pytest.mark.unit
class TestRemoteLoRAWrappedModule:
    
    def test_initialization(self):
        local_sub = nn.Linear(10, 10)
        comm = Mock()
        
        wrapped = RemoteLoRAWrappedModule(
            "test_module", local_sub, comm, "replace", True
        )
        
        assert wrapped.sub_name == "test_module"
        assert wrapped.local_sub == local_sub
        assert wrapped.comm == comm
        assert wrapped.combine_mode == "replace"
        assert wrapped.send_base_activations is True
    
    def test_forward_replace_mode(self):
        local_sub = nn.Linear(5, 5)
        comm = Mock()
        comm.lora_forward_with_base.return_value = [[0.5, 0.5, 0.5, 0.5, 0.5]]
        
        wrapped = RemoteLoRAWrappedModule(
            "test_module", local_sub, comm, "replace", True
        )
        
        x = torch.randn(1, 5)
        result = wrapped.forward(x)
        
        assert result.shape == (1, 5)
        comm.lora_forward_with_base.assert_called_once()
    
    def test_forward_add_delta_mode(self):
        local_sub = nn.Linear(5, 5)
        comm = Mock()
        comm.lora_forward_with_base.return_value = [[0.1, 0.1, 0.1, 0.1, 0.1]]
        
        wrapped = RemoteLoRAWrappedModule(
            "test_module", local_sub, comm, "add_delta", True
        )
        
        x = torch.randn(1, 5)
        result = wrapped.forward(x)
        
        assert result.shape == (1, 5)
        # Result should be base + delta
        comm.lora_forward_with_base.assert_called_once()
    
    def test_forward_without_base_activations(self):
        local_sub = nn.Linear(5, 5)
        comm = Mock()
        comm.lora_forward.return_value = [[0.5, 0.5, 0.5, 0.5, 0.5]]
        
        wrapped = RemoteLoRAWrappedModule(
            "test_module", local_sub, comm, "replace", False
        )
        
        x = torch.randn(1, 5)
        result = wrapped.forward(x)
        
        assert result.shape == (1, 5)
        comm.lora_forward.assert_called_once()
        comm.lora_forward_with_base.assert_not_called()
    
    def test_forward_no_output_error(self):
        local_sub = nn.Linear(5, 5)
        comm = Mock()
        comm.lora_forward_with_base.return_value = None
        
        wrapped = RemoteLoRAWrappedModule(
            "test_module", local_sub, comm, "replace", True
        )
        
        x = torch.randn(1, 5)
        
        with pytest.raises(RuntimeError, match="no output from A"):
            wrapped.forward(x)


@pytest.mark.unit
class TestBaseModelClient:
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_initialization(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.config.use_cache = True
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer.return_value = Mock()
        
        client = BaseModelClient(
            base_model="test_model",
            host_a="192.168.1.1",
            port_a=40000
        )
        
        assert len(client.comms) == 1
        assert client.comms[0].host_a == "192.168.1.1"
        assert client.comms[0].port_a == 40000
        assert client.combine_mode == "replace"
        assert client.use_optimization is True
        assert mock_model_instance.config.use_cache is False
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_multiple_contributors(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()
        
        contributors = [("host1", 30000), ("host2", 30001)]
        
        client = BaseModelClient(
            contributors=contributors,
            use_optimization=False
        )
        
        assert len(client.comms) == 2
        assert client.comms[0].host_a == "host1"
        assert client.comms[1].host_a == "host2"
        assert client.use_optimization is False
    
    def test_navigate_simple_path(self):
        client = Mock()
        client._navigate = BaseModelClient._navigate.__get__(client)
        
        # Create mock model structure
        model = Mock()
        model.transformer = Mock()
        
        result = client._navigate(model, ["transformer"])
        assert result == model.transformer
    
    def test_navigate_with_index(self):
        client = Mock()
        client._navigate = BaseModelClient._navigate.__get__(client)
        
        # Create mock model structure with indexable component
        model = Mock()
        model.transformer = Mock()
        model.transformer.h = [Mock(), Mock()]
        
        result = client._navigate(model, ["transformer", "h", "1"])
        assert result == model.transformer.h[1]
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_init_and_patch_success(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model_instance.transformer = Mock()
        mock_model_instance.transformer.h = [Mock()]
        mock_model_instance.transformer.h[0].attn = Mock()
        mock_model_instance.transformer.h[0].attn.c_attn = Mock()
        
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()
        
        client = BaseModelClient()
        
        # Mock communication
        comm = Mock()
        comm.init_request.return_value = ["transformer.h.0.attn.c_attn"]
        client.comms = [comm]
        
        client.init_and_patch()
        
        # Check that the module was patched
        assert hasattr(mock_model_instance.transformer.h[0].attn, 'c_attn')
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_init_and_patch_error_handling(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()
        
        client = BaseModelClient()
        
        # Mock communication with invalid module name
        comm = Mock()
        comm.init_request.return_value = ["nonexistent.module"]
        client.comms = [comm]
        
        # Should not raise exception, just print error
        client.init_and_patch()
        
        comm.init_request.assert_called_once()
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_forward_loss(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        
        # Mock output with loss
        mock_output = Mock()
        mock_output.loss = Mock()
        mock_output.loss.item.return_value = 0.5
        mock_model_instance.return_value = mock_output
        
        mock_model.return_value = mock_model_instance
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        client = BaseModelClient()
        
        loss = client.forward_loss("test text")
        
        assert loss == 0.5
        mock_tokenizer_instance.assert_called_once_with("test text", return_tensors="pt")
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_end_inference(self, mock_tokenizer, mock_model):
        mock_model_instance = Mock()
        mock_model_instance.config = Mock()
        mock_model.return_value = mock_model_instance
        mock_tokenizer.return_value = Mock()
        
        client = BaseModelClient()
        
        # Mock multiple communications
        comm1 = Mock()
        comm1.host_a = "host1"
        comm1.port_a = 30000
        comm1.end_inference.return_value = {"status": "complete"}
        
        comm2 = Mock()
        comm2.host_a = "host2"
        comm2.port_a = 30001
        comm2.end_inference.return_value = {"status": "complete"}
        
        client.comms = [comm1, comm2]
        
        client.end_inference()
        
        comm1.end_inference.assert_called_once()
        comm2.end_inference.assert_called_once() 