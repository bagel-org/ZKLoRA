# ZKLoRA Code Structure

This directory contains the core implementation of ZKLoRA. Here's an overview of the key components:

## Core Components

### Multi-Party Inference (MPI)
- `base_model_user_mpi/` - Client-side implementation for base model users (User B)
  - Handles remote LoRA module communication
  - Manages model patching and inference
- `lora_contributor_mpi/` - Server-side implementation for LoRA providers (User A)
  - Manages LoRA module serving
  - Handles proof generation requests

### Zero-Knowledge Proof Generation
- `zk_proof_generator.py` - Core proof generation and verification logic
  - Generates zero-knowledge proofs for LoRA computations
  - Provides batch verification capabilities
- `mpi_lora_onnx_exporter.py` - ONNX export utilities for proof generation
  - Converts LoRA modules to ONNX format
  - Prepares models for zero-knowledge circuit compilation

### Merkle Tree Implementation
- `libs/merkle/` - Rust implementation of Merkle tree functionality
  - Provides efficient hashing and tree construction
  - Exposes Python bindings via PyO3
- `activations_commit.py` - Python interface for Merkle tree operations
  - Handles model activation commitments
  - Manages proof verification data

## File Structure

```
zklora/
├── __init__.py                 # Package exports
├── activations_commit.py       # Merkle tree interface
├── base_model_user_mpi/        # Client implementation
├── lora_contributor_mpi/       # Server implementation
├── libs/
│   └── merkle/                # Rust Merkle tree implementation
├── mpi_lora_onnx_exporter.py  # ONNX export utilities
└── zk_proof_generator.py      # Proof generation core
```

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

## Implementation Details

- The system uses a client-server architecture for multi-party inference
- Zero-knowledge proofs are generated using the EZKL framework
- Merkle trees are implemented in Rust for performance
- ONNX export is used to prepare models for proof generation
- All network communication is handled via TCP sockets

For usage examples, see the sample scripts in the main README.md. 