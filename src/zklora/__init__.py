__version__ = '0.1.2'

from .zk_proof_generator import batch_verify_proofs
from .zk_proof_generator_optimized import (
    generate_proofs_optimized_parallel,
    batch_verify_proofs_optimized
)
from .lora_contributor_mpi import LoRAServer, LoRAServerSocket
from .base_model_user_mpi import BaseModelClient
from .polynomial_commit import commit_activations, verify_commitment


__all__ = [
    'batch_verify_proofs',
    'batch_verify_proofs_optimized',
    'generate_proofs_optimized_parallel',
    'LoRAServer',
    'LoRAServerSocket',
    'BaseModelClient',
    'commit_activations',
    'verify_commitment',
    '__version__',
]