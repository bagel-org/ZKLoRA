"""
Zero-knowledge proof generator for LoRA modules using Halo2.
"""
from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .halo2_wrapper import Halo2Prover

logger = logging.getLogger(__name__)

async def generate_proofs(
    onnx_dir: Union[str, Path],
    json_dir: Union[str, Path],
    output_dir: Union[str, Path],
    verbose: bool = False
) -> bool:
    """Generate proofs for all ONNX models in the directory."""
    onnx_dir = Path(onnx_dir)
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ONNX models
    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        logger.warning(f"No ONNX files found in {onnx_dir}")
        return False

    # Process each model
    for onnx_file in onnx_files:
        model_name = onnx_file.stem
        json_file = json_dir / f"{model_name}.json"
        if not json_file.exists():
            logger.warning(f"JSON file not found for {model_name}")
            continue

        # Load parameters
        with open(json_file) as f:
            params = json.load(f)

        # Initialize proof generator
        generator = ZKProofGenerator(
            onnx_model_path=onnx_file,
            out_dir=output_dir
        )

        # Generate proof
        success, proof_path = await generator.generate_proof(
            input_data=params["input"],
            weight_a=params["weight_a"],
            weight_b=params["weight_b"],
            proof_id=model_name
        )

        if verbose:
            if success:
                logger.info(f"Generated proof for {model_name}")
            else:
                logger.error(f"Failed to generate proof for {model_name}")

    return True

def resolve_proof_paths(
    proof_dir: Union[str, Path],
    proof_ids: Optional[List[str]] = None
) -> List[Path]:
    """Resolve proof paths from proof IDs."""
    proof_dir = Path(proof_dir)
    if proof_ids is None:
        # Find all proof files
        return list(proof_dir.glob("*.proof"))
    else:
        # Find specific proof files
        return [proof_dir / f"{pid}.proof" for pid in proof_ids]

class ZKProofGenerator:
    """Generates zero-knowledge proofs for LoRA modules using Halo2."""
    
    def __init__(self, 
                 onnx_model_path: Path,
                 settings_path: Optional[Path] = None,
                 out_dir: Optional[Path] = None):
        """Initialize the proof generator."""
        self.onnx_model_path = onnx_model_path
        self.out_dir = out_dir or Path("proofs")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Halo2 prover
        self.prover = Halo2Prover(settings_path)
        
        # Load ONNX model for input/output validation
        self.session = ort.InferenceSession(str(onnx_model_path))
        
        # Extract model shapes
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        # Generate settings if not provided
        if settings_path is None:
            self.settings = self.prover.gen_settings(
                input_shape=self.input_shape,
                output_shape=self.output_shape
            )
            settings_file = self.out_dir / "settings.json"
            self.prover.save_settings(settings_file)
            
    def _validate_shapes(self, 
                        input_data: np.ndarray,
                        weight_a: np.ndarray,
                        weight_b: np.ndarray) -> None:
        """Validate matrix shapes for compatibility."""
        if input_data.shape[1] != weight_a.shape[0]:
            raise ValueError(
                f"Input shape {input_data.shape} incompatible with "
                f"weight_a shape {weight_a.shape}"
            )
        if weight_a.shape[1] != weight_b.shape[0]:
            raise ValueError(
                f"Weight_a shape {weight_a.shape} incompatible with "
                f"weight_b shape {weight_b.shape}"
            )
            
    async def generate_proof(self,
                           input_data: Union[np.ndarray, List],
                           weight_a: Union[np.ndarray, List],
                           weight_b: Union[np.ndarray, List],
                           proof_id: str) -> Tuple[bool, Path]:
        """Generate a proof for a LoRA module."""
        # Convert inputs to numpy arrays
        input_data = np.asarray(input_data)
        weight_a = np.asarray(weight_a)
        weight_b = np.asarray(weight_b)
        
        # Validate shapes
        self._validate_shapes(input_data, weight_a, weight_b)
        
        # Generate witness
        witness = self.prover.gen_witness(input_data, weight_a, weight_b)
        
        # Run mock verification
        if not self.prover.mock(witness):
            raise ValueError("Mock verification failed")
            
        # Generate proof
        proof_path = self.out_dir / f"{proof_id}.proof"
        success = await self.prover.prove(witness, proof_path)
        
        if not success:
            logger.error(f"Failed to generate proof for {proof_id}")
            return False, proof_path
            
        return True, proof_path
        
    async def verify_proof(self,
                          proof_path: Path,
                          public_inputs: Optional[Dict] = None) -> bool:
        """Verify a proof."""
        return await self.prover.verify(proof_path, public_inputs=public_inputs)
        
    async def batch_verify_proofs(self,
                                proof_paths: List[Path],
                                public_inputs: Optional[List[Dict]] = None) -> List[bool]:
        """Verify multiple proofs in parallel."""
        if public_inputs is None:
            public_inputs = [None] * len(proof_paths)
            
        verify_tasks = [
            self.verify_proof(path, inputs)
            for path, inputs in zip(proof_paths, public_inputs)
        ]
        
        return await asyncio.gather(*verify_tasks)
        
    def get_proof_path(self, proof_id: str) -> Path:
        """Get the path for a proof file."""
        return self.out_dir / f"{proof_id}.proof"
