"""
Halo2 wrapper for ZKLoRA that replaces EZKL functionality.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .libs.zklora_halo2 import generate_proof, verify_proof

def flatten_matrix(matrix: List[List[float]]) -> List[float]:
    """Flatten a 2D matrix into a 1D list."""
    return [x for row in matrix for x in row]

class Halo2Prover:
    """Handles proof generation and verification using Halo2."""
    
    def __init__(self, settings_path: Optional[Path] = None):
        """Initialize the prover with optional settings."""
        self.settings = {}
        if settings_path is not None:
            with open(settings_path) as f:
                settings = json.load(f)
                # Convert shape lists back to tuples
                settings["input_shape"] = tuple(settings["input_shape"])
                settings["output_shape"] = tuple(settings["output_shape"])
                self.settings = settings
                
    def gen_settings(self, 
                    input_shape: Tuple[int, ...],
                    output_shape: Tuple[int, ...],
                    scale: float = 1e4) -> Dict:
        """Generate circuit settings for the given shapes."""
        settings = {
            "input_shape": input_shape,
            "output_shape": output_shape,
            "scale": scale,
            "bits": 32,  # Default to 32-bit precision
            "public_inputs": ["input", "output"],
            "private_inputs": ["weight_a", "weight_b"]
        }
        self.settings = settings
        return settings
    
    def save_settings(self, path: Path) -> None:
        """Save settings to a JSON file."""
        settings = self.settings.copy()
        # Convert tuples to lists for JSON serialization
        settings["input_shape"] = list(settings["input_shape"])
        settings["output_shape"] = list(settings["output_shape"])
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
            
    def compile_circuit(self, 
                       onnx_path: Path,
                       settings: Optional[Dict] = None) -> None:
        """
        Compile the circuit from ONNX model.
        This is a no-op in Halo2 as we generate the circuit dynamically.
        """
        if settings is not None:
            self.settings = settings
            
    def gen_witness(self,
                    input_data: Union[np.ndarray, List],
                    weight_a: Union[np.ndarray, List],
                    weight_b: Union[np.ndarray, List]) -> Dict:
        """Generate witness data for the circuit."""
        # Convert inputs to numpy arrays if needed
        input_data = np.asarray(input_data)
        weight_a = np.asarray(weight_a)
        weight_b = np.asarray(weight_b)
        
        # Scale values according to settings
        scale = self.settings.get("scale", 1e4)
        input_scaled = (input_data * scale).astype(np.int64)
        weight_a_scaled = (weight_a * scale).astype(np.int64)
        weight_b_scaled = (weight_b * scale).astype(np.int64)
        
        # Compute expected output
        output = input_data @ weight_a @ weight_b
        output_scaled = (output * scale).astype(np.int64)
        
        return {
            "input": input_scaled.tolist(),
            "weight_a": weight_a_scaled.tolist(),
            "weight_b": weight_b_scaled.tolist(),
            "output": output_scaled.tolist()
        }
        
    async def prove(self,
                   witness: Dict,
                   proof_path: Path,
                   settings: Optional[Dict] = None) -> bool:
        """Generate a proof using the witness data."""
        if settings is not None:
            self.settings = settings
            
        # Generate proof using Rust implementation
        proof_data = generate_proof(
            flatten_matrix(witness["input"]),
            flatten_matrix(witness["weight_a"]),
            flatten_matrix(witness["weight_b"])
        )
        
        # Save proof to file
        with open(proof_path, 'wb') as f:
            # Ensure proof_data is bytes
            if isinstance(proof_data, list):
                proof_data = bytes(proof_data)
            f.write(proof_data)
            
        return True
        
    async def verify(self,
                    proof_path: Path,
                    settings: Optional[Dict] = None,
                    public_inputs: Optional[Dict] = None) -> bool:
        """Verify a proof."""
        if settings is not None:
            self.settings = settings
            
        # Read proof from file
        with open(proof_path, 'rb') as f:
            proof_data = f.read()
            
        # Get public inputs
        if public_inputs is None:
            public_inputs = []
            
        # Verify using Rust implementation
        return verify_proof(proof_data, public_inputs)
        
    def mock(self,
            witness: Dict,
            settings: Optional[Dict] = None) -> bool:
        """Mock verification for testing."""
        if settings is not None:
            self.settings = settings
            
        # Simple check that shapes match
        input_shape = np.array(witness["input"]).shape
        weight_a_shape = np.array(witness["weight_a"]).shape
        weight_b_shape = np.array(witness["weight_b"]).shape
        output_shape = np.array(witness["output"]).shape
        
        expected_output_shape = (input_shape[0], weight_b_shape[1])
        return output_shape == expected_output_shape 