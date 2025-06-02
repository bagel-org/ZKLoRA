from __future__ import annotations

import pytest
import tempfile
import os
import time
import threading
import socket
import torch
import numpy as np
import asyncio
from unittest.mock import patch, Mock, AsyncMock

from zklora import (
    LoRAServer,
    LoRAServerSocket,
    BaseModelClient
)
from zklora.low_rank_circuit import export_optimized_lora_circuit
from zklora.zk_proof_generator_optimized import (
    generate_proofs_optimized_parallel,
    batch_verify_proofs_optimized
)


@pytest.mark.integration
@pytest.mark.slow
class TestFullWorkflow:
    
    @pytest.fixture
    def setup_server_client(self):
        server_port = 40000
        server_host = "127.0.0.1"
        stop_event = threading.Event()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yield {
                'host': server_host,
                'port': server_port,
                'stop_event': stop_event,
                'tmpdir': tmpdir
            }
            
            stop_event.set()
    
    def wait_for_port(self, host, port, timeout=5):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    return True
                time.sleep(0.1)
            except:
                time.sleep(0.1)
        return False
    
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('peft.PeftModel.from_pretrained')
    def test_optimized_lora_workflow(
        self, mock_peft, mock_tokenizer, mock_model, setup_server_client
    ):
        config = setup_server_client
        
        mock_base_model = Mock()
        mock_base_model.config = Mock()
        mock_base_model.config.use_cache = False
        mock_model.return_value = mock_base_model
        
        mock_lora_module = Mock()
        mock_lora_A = Mock()
        mock_lora_B = Mock()
        mock_lora_A.weight = torch.randn(768, 16)
        mock_lora_B.weight = torch.randn(16, 768)
        mock_lora_module.lora_A = {'default': mock_lora_A}
        mock_lora_module.lora_B = {'default': mock_lora_B}
        
        # Mock named_parameters to return an iterable with lora parameters
        mock_lora_param = Mock()
        mock_lora_param.name = "lora_A.weight"
        mock_lora_module.named_parameters.return_value = [("lora_A.weight", mock_lora_param)]
        
        mock_peft_model = Mock()
        mock_peft_model.named_modules.return_value = [
            ('base_model.model.transformer.h.0.attn.c_attn', mock_lora_module)
        ]
        mock_peft.return_value = mock_peft_model
        
        server = LoRAServer(
            base_model_name="test_model",
            lora_model_id="test_lora",
            out_dir=config['tmpdir'],
            use_optimization=True
        )
        
        server_thread = LoRAServerSocket(
            config['host'], 
            config['port'], 
            server, 
            config['stop_event']
        )
        server_thread.daemon = True
        server_thread.start()
        
        assert self.wait_for_port(config['host'], config['port'])
        
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        mock_output = Mock()
        mock_output.loss = Mock()
        mock_output.loss.item.return_value = 0.5
        mock_base_model.return_value = mock_output
        
        client = BaseModelClient(
            base_model="test_model",
            contributors=[(config['host'], config['port'])],
            use_optimization=True
        )
        
        injection_points = server.list_lora_injection_points()
        assert len(injection_points) > 0
        
        x_data = np.random.randn(1, 128, 768)
        x_tensor = torch.from_numpy(x_data).float()
        base_activations = np.random.randn(1, 128, 768)
        
        output = server.apply_lora(
            injection_points[0], 
            x_tensor, 
            base_activations
        )
        assert output is not None
        
        config['stop_event'].set()
        server_thread.join(timeout=2)


@pytest.mark.integration
def test_low_rank_circuit_export_and_load():
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
        
        assert os.path.exists(os.path.join(tmpdir, "test_module_optimized.onnx"))
        assert os.path.exists(os.path.join(tmpdir, "test_module_config.json"))
        assert os.path.exists(os.path.join(tmpdir, "test_module_lookup.json"))
        
        assert config['optimizations']['low_rank_factorization'] is True
        assert config['optimizations']['rank'] == 16
        assert config['performance_gains']['total_speedup'] == 491520


@pytest.mark.integration
@pytest.mark.asyncio
async def test_proof_generation_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(2):
            A = torch.randn(64, 8)
            B = torch.randn(8, 64)
            x_data = np.random.randn(1, 32, 64)
            base_activations = np.random.randn(1, 32, 64)
            
            export_optimized_lora_circuit(
                f"module_{i}",
                A, B, x_data, base_activations,
                tmpdir,
                verbose=False
            )
        
        with patch('zklora.zk_proof_generator_optimized.ezkl') as mock_ezkl:
            mock_ezkl.prove.return_value = True
            mock_ezkl.gen_witness = AsyncMock(return_value=None)
            
            with patch('os.path.isfile', return_value=False):
                result = await generate_proofs_optimized_parallel(
                    onnx_dir=tmpdir,
                    json_dir=tmpdir,
                    output_dir=tmpdir,
                    max_workers=2,
                    verbose=False
                )
        
        assert result is not None


@pytest.mark.integration
def test_quantization_accuracy():
    from zklora.low_rank_circuit import LowRankQuantizer
    
    quantizer = LowRankQuantizer(weight_bits=4, activation_bits=8)
    
    original_A = torch.randn(768, 16) * 0.1
    original_B = torch.randn(16, 768) * 0.1
    
    A_q, B_q, quant_params = quantizer.quantize_weights(original_A, original_B)
    
    A_reconstructed = torch.from_numpy(A_q).float() * quant_params['A_scale']
    B_reconstructed = torch.from_numpy(B_q).float() * quant_params['B_scale']
    
    A_error = torch.mean(torch.abs(original_A - A_reconstructed)).item()
    B_error = torch.mean(torch.abs(original_B - B_reconstructed)).item()
    
    assert A_error < 0.02
    assert B_error < 0.02
    
    rel_A_error = A_error / torch.mean(torch.abs(original_A)).item()
    rel_B_error = B_error / torch.mean(torch.abs(original_B)).item()
    
    assert rel_A_error < 0.25
    assert rel_B_error < 0.25


@pytest.mark.integration
def test_lookup_table_consistency():
    from zklora.low_rank_circuit import LookupTableCircuit
    
    for weight_bits in [2, 4]:
        for activation_bits in [4, 8]:
            circuit = LookupTableCircuit(
                weight_bits=weight_bits, 
                activation_bits=activation_bits
            )
            
            for _ in range(100):
                w = np.random.randint(
                    -(2**(weight_bits-1)), 
                    2**(weight_bits-1)
                )
                a = np.random.randint(0, 2**activation_bits)
                
                lookup_result = circuit.lookup_table[(w, a)]
                expected = w * a
                
                assert lookup_result == expected 