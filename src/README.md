# ZKLoRA Source Code Structure

This directory contains the core implementation of ZKLoRA. Here's a detailed overview of the key components and their interactions.

## Directory Structure

```
src/
├── zklora/                     # Main package directory
│   ├── __init__.py            # Package exports
│   ├── activations_commit.py   # Merkle tree interface
│   ├── base_model_user_mpi/    # Client implementation (User B)
│   ├── lora_contributor_mpi/   # Server implementation (User A)
│   ├── libs/
│   │   └── merkle/            # Rust Merkle tree implementation
│   ├── mpi_lora_onnx_exporter.py  # ONNX export utilities
│   └── zk_proof_generator.py   # Proof generation core
├── scripts/                    # Sample usage scripts
├── pyproject.toml             # Build configuration
└── requirements.txt           # Dependencies
```

## Implementation Details

### Zero-Knowledge Architecture

The zero-knowledge proof system in ZKLoRA is built on polynomial commitments and succinct proofs. The `zk_proof_generator.py` module orchestrates the proof generation process by:

1. Converting LoRA modules to ONNX format using `mpi_lora_onnx_exporter.py`
2. Computing Merkle roots of model activations via `activations_commit.py`
3. Generating zero-knowledge proofs that validate LoRA compatibility

### Multi-Party Inference Protocol

The MPI system enables secure interaction between the base model user (B) and LoRA provider (A) through:

- Encrypted communication channels for activation exchange
- Asynchronous proof generation that doesn't block inference
- Efficient state management for handling multiple concurrent sessions

The `base_model_user_mpi` and `lora_contributor_mpi` directories contain the client and server implementations respectively, with careful attention to thread safety and resource management.

### Merkle Tree Implementation

The Merkle tree system, implemented in Rust for performance, provides:

- Fast commitment generation for model activations
- Efficient proof verification
- Compact representation of large activation tensors

The Rust implementation is wrapped with Python bindings in the `libs/merkle` directory.

### Performance Considerations

ZKLoRA achieves its 1-2 second verification time through:

- Parallel proof generation for multiple LoRA modules
- Optimized ONNX conversions that minimize computational overhead
- Efficient Merkle tree implementations in Rust
- Careful memory management during large model operations

For detailed usage examples and high-level architecture, please refer to the [main README](../../README.md) in the project root.

## Core Components

### Multi-Party Inference (MPI)
- `base_model_user_mpi/`: Client-side implementation for base model users (User B)
  - Handles remote LoRA module communication
  - Manages model patching and inference
- `lora_contributor_mpi/`: Server-side implementation for LoRA providers (User A)
  - Manages LoRA module serving
  - Handles proof generation requests

### Zero-Knowledge Components
- `zk_proof_generator.py`: Core proof generation and verification
- `mpi_lora_onnx_exporter.py`: ONNX export utilities for proof generation
- `activations_commit.py`: Merkle tree interface for model commitments

### Build & Distribution
- `pyproject.toml`: Package metadata and build configuration
- `requirements.txt`: Project dependencies

### Sample Scripts
- `scripts/base_model_user_sample_script.py`: Example client usage
- `scripts/lora_contributor_sample_script.py`: Example server usage
- `scripts/verify_proofs.py`: Proof verification utility

## Key Interfaces

1. **Base Model User (B)**
```python
from zklora import BaseModelClient

client = BaseModelClient(base_model="distilgpt2")
client.init_and_patch()
loss = client.forward_loss("input text")
```

2. **LoRA Provider (A)**
```python
from zklora import LoRAServer

server = LoRAServer(base_model_name="distilgpt2", 
                   lora_model_id="path/to/lora")
server.list_lora_injection_points()
```

3. **Proof Verification**
```python
from zklora import batch_verify_proofs

verify_time, num_proofs = batch_verify_proofs(
    proof_dir="proof_artifacts"
)
```

For detailed implementation information, please refer to the individual module documentation. 