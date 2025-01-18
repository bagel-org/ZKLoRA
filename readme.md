<p align="center">
  <img src="paper/figs/bagel-logo.png" alt="Bagel Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/bagel-org/zklora">
    <img src="https://img.shields.io/github/stars/bagelopenai/zklora?style=social&scale=2" width="150" alt="GitHub stars"/>
  </a>
</p>

<p align="center">
  <a href="https://twitter.com/bagelopenAI">
    <img src="https://img.shields.io/twitter/follow/bagelopenAI?style=social" alt="Twitter Follow"/>
  </a>
  
  <a href="https://blog.bagel.net">
    <img src="https://img.shields.io/badge/Follow%20on-Substack-orange?style=social&logo=substack" alt="Substack Follow"/>
  </a>
</p>

## ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification

## Table of Contents
- [ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification](#zklora-efficient-zero-knowledge-proofs-for-lora-verification)
  - [Key Performance Results](#key-performance-results)
  - [Multi-Party Inference (MPI) Architecture](#multi-party-inference-mpi-architecture)
- [Quick Usage Instructions](#quick-usage-instructions)
  - [1. LoRA Provider Side (User A)](#1-lora-provider-side-user-a)
  - [2. Base Model User Side (User B)](#2-base-model-user-side-user-b)
  - [3. Proof Verification](#3-proof-verification)
- [Summary](#summary)
- [Credits](#credits)
- [License](#license)

Low-Rank Adaptation (LoRA) is a widely adopted method for customizing large-scale language models. In distributed, untrusted training environments, an open source base model user may want to use LoRA weights created by an external contributor, leading to two requirements:

1. **Base Model User Verification**: The user must confirm that the LoRA weights are effective when paired with the intended base model.
2. **LoRA Contributor Protection**: The contributor must keep their proprietary LoRA weights private until compensation is assured.

**ZKLoRA** is a zero-knowledge verification protocol that relies on polynomial commitments, succinct proofs, and multi-party inference to verify LoRA–base model compatibility without exposing LoRA weights.

### Key Performance Results

Our benchmarks show:
- Verification time of 1-2 seconds per LoRA module
- Practical scaling with number of LoRA modules (e.g., 80+ modules for 70B parameter models)
- Efficient handling of varying LoRA sizes (from 24K to 327K parameters per module)

### Multi-Party Inference (MPI) Architecture

In our multi-party inference scenario:
- **User A** (LoRA contributor) holds LoRA-augmented submodules
- **User B** (base model user) has the large base model
- They collaborate on inference while keeping LoRA computations hidden
- **A** generates zero-knowledge proofs of computation correctness
- **B** can verify these proofs offline using provided artifacts

## Quick Usage Instructions

### 1. LoRA Provider Side (User A)

Use `lora_contributor_sample_script.py` to:
- Host LoRA submodules
- Handle inference requests
- Generate proof artifacts

```python
import argparse
import threading
import time

from zklora import LoRAServer, AServerTCP

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--lora_model_id", default="ng0-k1/distilgpt2-finetuned-es")
    parser.add_argument("--out_dir", default="a-out")
    args = parser.parse_args()

    stop_event = threading.Event()
    server_obj = LoRAServer(args.base_model, args.lora_model_id, args.out_dir)
    t = AServerTCP(args.host, args.port_a, server_obj, stop_event)
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[A-Server] stopping.")
    stop_event.set()
    t.join()

if __name__ == "__main__":
    main()
```

### 2. Base Model User Side (User B)

Use `base_model_user_sample_script.py` to:
- Load and patch the base model
- Connect to A's submodules
- Perform inference
- Trigger proof generation

```python
import argparse

from zklora import BaseModelClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_a", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--combine_mode", choices=["replace","add_delta"], default="add_delta")
    args = parser.parse_args()

    client = BaseModelClient(args.base_model, args.host_a, args.port_a, args.combine_mode)
    client.init_and_patch()

    # Run inference => triggers remote LoRA calls on A
    text = "Hello World, this is a LoRA test."
    loss_val = client.forward_loss(text)
    print(f"[B] final loss => {loss_val:.4f}")

    # End inference => A finalizes proofs offline
    client.end_inference()
    print("[B] done. B can now fetch proof files from A and verify them offline.")

if __name__=="__main__":
    main()
```

### 3. Proof Verification

Use `verify_proofs.py` to validate the proof artifacts:

```python
#!/usr/bin/env python3
"""
Verify LoRA proof artifacts in a given directory.

Example usage:
  python verify_proofs.py --proof_dir a-out --verbose
"""

import argparse
from zklora import batch_verify_proofs

def main():
    parser = argparse.ArgumentParser(
        description="Verify LoRA proof artifacts in a given directory."
    )
    parser.add_argument(
        "--proof_dir",
        type=str,
        default="proof_artifacts",
        help="Directory containing proof files (.pf), plus settings, vk, srs."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more details during verification."
    )
    args = parser.parse_args()

    total_verify_time, num_proofs = batch_verify_proofs(
        proof_dir=args.proof_dir,
        verbose=args.verbose
    )
    print(f"Done verifying {num_proofs} proofs. Total time: {total_verify_time:.2f}s")

if __name__ == "__main__":
    main()
```

## Summary

- **ZKLoRA** enables trust-minimized LoRA verification through zero-knowledge proofs
- Achieves **1-2 second verification** per module, even for billion-parameter models
- Supports **multi-party inference** with secure activation exchange
- Maintains **complete privacy** of LoRA weights while ensuring compatibility
- Scales efficiently to handle multiple LoRA modules in production environments

Future work includes adding polynomial commitments for base model activations and supporting multi-contributor LoRA scenarios.

## Credits

ZKLoRA builds upon several excellent open source libraries:

- **[PEFT](https://github.com/huggingface/peft)**: Parameter-Efficient Fine-Tuning library by Hugging Face
- **[Transformers](https://github.com/huggingface/transformers)**: State-of-the-art Natural Language Processing by Hugging Face
- **[dusk-merkle](https://github.com/dusk-network/dusk-merkle)**: Merkle tree implementation in Rust
- **[BLAKE3](https://github.com/BLAKE3-team/BLAKE3)**: Cryptographic hash function
- **[EZKL](https://github.com/zkonduit/ezkl)**: Zero-knowledge proof system for neural networks
- **[ONNX Runtime](https://github.com/microsoft/onnxruntime)**: Cross-platform ML model inference

We are grateful to the maintainers and contributors of these projects for their valuable work.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License - see the [LICENSE](LICENSE) file for details. This means you are free to use, share, and adapt the work for non-commercial purposes, as long as you give appropriate credit and distribute your contributions under the same license.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)