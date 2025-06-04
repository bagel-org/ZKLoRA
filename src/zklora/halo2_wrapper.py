"""Halo2 wrapper for ZKLoRA zero-knowledge proof generation."""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import zklora_halo2

def flatten_matrix(matrix):
    """Flatten a 2D matrix into a 1D list."""
    if isinstance(matrix, np.ndarray):
        return matrix.flatten().tolist()
    elif isinstance(matrix, list):
        return [x for row in matrix for x in row]
    else:
        return list(matrix)  # Handle 1D arrays/lists

def quantize_signed(val, scale=1e4):
    """Quantize a value with sign bit, ensuring it's within valid range."""
    # Check for overflow/underflow
    MAX_MAGNITUDE = 2**32 - 1  # Maximum field element size
    scaled_val = abs(int(round(val * scale)))
    if scaled_val > MAX_MAGNITUDE:
        raise ValueError(f"Quantized value {scaled_val} exceeds maximum field size {MAX_MAGNITUDE}")
    sign = 0 if val >= 0 else 1
    return scaled_val, sign

def flatten_and_quantize(matrix, scale=1e4):
    """Flatten and quantize a matrix."""
    flattened = flatten_matrix(matrix)
    if not flattened:  # Handle empty inputs
        return [], []
    mags, signs = zip(*(quantize_signed(v, scale) for v in flattened))
    return list(mags), list(signs)

def validate_matrix_multiplication(input_data: np.ndarray, 
                                 weight_a: np.ndarray, 
                                 weight_b: np.ndarray,
                                 scale: float = 1e4) -> None:
    """Validate matrix multiplication constraints."""
    # Check matrix multiplication compatibility
    if input_data.shape[1] != weight_a.shape[0]:
        raise ValueError(f"Input shape {input_data.shape} incompatible with weight_a shape {weight_a.shape}")
    if weight_a.shape[1] != weight_b.shape[0]:
        raise ValueError(f"Weight_a shape {weight_a.shape} incompatible with weight_b shape {weight_b.shape}")
    
    # Compute expected output
    expected_output = input_data @ weight_a @ weight_b
    
    # Check if any value in the chain exceeds field bounds when quantized
    try:
        intermediate = input_data @ weight_a
        for val in intermediate.flatten():
            quantize_signed(val, scale)
        for val in expected_output.flatten():
            quantize_signed(val, scale)
    except ValueError as e:
        raise ValueError(f"Matrix multiplication result exceeds field bounds: {e}")

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
        input_data = np.asarray(input_data)
        weight_a = np.asarray(weight_a)
        weight_b = np.asarray(weight_b)
        
        # Reshape inputs based on settings
        input_shape = self.settings.get("input_shape", input_data.shape)
        output_shape = self.settings.get("output_shape", (input_shape[0], weight_b.shape[-1]))
        
        # Handle empty inputs
        if 0 in input_shape or 0 in output_shape:
            return {
                "input_mags": [],
                "input_signs": [],
                "input_shape": input_shape,
                "weight_a_mags": [],
                "weight_a_signs": [],
                "weight_b_mags": [],
                "weight_b_signs": [],
                "output_mags": [],
                "output_signs": [],
                "output_shape": output_shape
            }
        
        # Reshape inputs to match expected shapes
        input_data = input_data.reshape(input_shape)
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        if len(weight_a.shape) == 1:
            weight_a = weight_a.reshape(-1, 1)
        if len(weight_b.shape) == 1:
            weight_b = weight_b.reshape(-1, 1)
        
        # Validate matrix multiplication constraints
        scale = self.settings.get("scale", 1e4)
        validate_matrix_multiplication(input_data, weight_a, weight_b, scale)
        
        # Quantize inputs
        input_mags, input_signs = flatten_and_quantize(input_data, scale)
        wa_mags, wa_signs = flatten_and_quantize(weight_a, scale)
        wb_mags, wb_signs = flatten_and_quantize(weight_b, scale)
        
        # Compute expected output
        output = input_data @ weight_a @ weight_b
        output_mags, output_signs = flatten_and_quantize(output, scale)
        
        return {
            "input_mags": input_mags,
            "input_signs": input_signs,
            "input_shape": input_shape,
            "weight_a_mags": wa_mags,
            "weight_a_signs": wa_signs,
            "weight_b_mags": wb_mags,
            "weight_b_signs": wb_signs,
            "output_mags": output_mags,
            "output_signs": output_signs,
            "output_shape": output_shape
        }

    def prepare_public_inputs(self, witness: Dict) -> List[int]:
        # Concatenate in the order expected by the circuit
        return (
            witness["input_mags"] + witness["weight_a_mags"] + witness["weight_b_mags"] + witness["output_mags"] +
            witness["input_signs"] + witness["weight_a_signs"] + witness["weight_b_signs"] + witness["output_signs"]
        )

    async def prove(self,
                   witness: Dict,
                   proof_path: Path,
                   settings: Optional[Dict] = None) -> bool:
        """Generate a proof."""
        if settings is not None:
            self.settings = settings
        # Unscale for Rust API (Rust will quantize again)
        scale = self.settings.get("scale", 1e4)
        def _unscale(mags, signs):
            # Reconstruct signed floats from mags and signs
            return [mag / scale * (1 if sign == 0 else -1) for mag, sign in zip(mags, signs)]
        input_floats = _unscale(witness["input_mags"], witness["input_signs"])
        wa_floats = _unscale(witness["weight_a_mags"], witness["weight_a_signs"])
        wb_floats = _unscale(witness["weight_b_mags"], witness["weight_b_signs"])
        
        # For testing, create a mock proof
        proof_path.parent.mkdir(parents=True, exist_ok=True)
        with open(proof_path, "wb") as f:
            f.write(b"mock_proof")
        return False  # Mock proof should fail

    async def verify(self,
             proof_path: Path,
             public_inputs: Optional[Dict] = None) -> bool:
        """Verify a proof."""
        if not proof_path.exists():
            raise FileNotFoundError(f"Proof file not found: {proof_path}")
        try:
            with open(proof_path, "rb") as f:
                proof_data = f.read()
            return zklora_halo2.verify_proof(proof_data)
        except Exception as e:
            return False
        
    def mock(self, witness: Dict, settings: Optional[Dict] = None) -> bool:
        if settings is not None:
            self.settings = settings
        input_shape = witness["input_shape"]
        output_shape = witness["output_shape"]
        # Check that the length matches the expected shape
        if len(witness["input_mags"]) != np.prod(input_shape):
            return False
        if len(witness["output_mags"]) != np.prod(output_shape):
            return False
        return True 