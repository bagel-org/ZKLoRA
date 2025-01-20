<p align="center">
  <img src="bagel-logo.png" alt="Bagel Logo" width="200"/>
</p>

<p align="center">
  <a href="https://twitter.com/bagelopenAI">
    <img src="https://img.shields.io/twitter/follow/bagelopenAI?style=flat-square" alt="Twitter Follow"/>
  </a>
  
  <a href="https://blog.bagel.net">
    <img src="https://img.shields.io/badge/Follow%20on-Substack-orange?style=flat-square&logo=substack" alt="Substack Follow"/>
  </a>
  
  <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square" alt="License"/>
  </a>
</p>

<h1 align="center">ZKLoRA</h1>
<h3 align="center">Efficient Zero-Knowledge Proofs for LoRA Verification</h3>

<hr>

## ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification

Low-Rank Adaptation (LoRA) is a widely adopted method for customizing large-scale language models. In distributed, untrusted training environments, an open source base model user may want to use LoRA weights created by an external contributor, leading to two requirements:

1. **Base Model User Verification**: The user must confirm that the LoRA weights are effective when paired with the intended base model.
2. **LoRA Contributor Protection**: The contributor must keep their proprietary LoRA weights private until compensation is assured.

To solve this, we created **ZKLoRA** a zero-knowledge verification protocol that relies on polynomial commitments, succinct proofs, and multi-party inference to verify LoRA–base model compatibility without exposing LoRA weights. With ZKLoRA, verification of LoRA modules takes just 1-2 seconds, even for state-of-the-art language models with tens of billions of parameters.

For detailed information about this research, please refer to our paper: `paper.pdf`

<h2 align="center">Quick Usage Instructions</h2>

### 1. LoRA Contributor Side (User A)

First, install ZKLoRA using pip:
```bash
pip install zklora
```

Use `src/scripts/lora_contributor_sample_script.py` to:
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

Use `src/scripts/base_model_user_sample_script.py` to:
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

Use `src/scripts/verify_proofs.py` to validate the proof artifacts:

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

<hr>

<h2 align="center">Code Structure</h2>

For detailed information about the codebase organization and implementation details, see [Code Structure](src/zklora/README.md).

<h2 align="center">Summary</h2>

<table>
<tr>
<td>✓</td><td><strong>Trust-Minimized Verification:</strong> Zero-knowledge proofs enable secure LoRA validation</td>
</tr>
<tr>
<td>✓</td><td><strong>Rapid Verification:</strong> 1-2 second processing per module, even for billion-parameter models</td>
</tr>
<tr>
<td>✓</td><td><strong>Multi-Party Inference:</strong> Protected activation exchange between parties</td>
</tr>
<tr>
<td>✓</td><td><strong>Complete Privacy:</strong> LoRA weights remain confidential while ensuring compatibility</td>
</tr>
<tr>
<td>✓</td><td><strong>Production Ready:</strong> Efficiently scales to handle multiple LoRA modules</td>
</tr>
</table>

Future work includes adding polynomial commitments for base model activations and supporting multi-contributor LoRA scenarios.

<h2 align="center">Credits</h2>

ZKLoRA is built upon these outstanding open source projects:

| Project | Description |
|---------|-------------|
| [PEFT](https://github.com/huggingface/peft) | Parameter-Efficient Fine-Tuning library by Hugging Face |
| [Transformers](https://github.com/huggingface/transformers) | State-of-the-art Natural Language Processing |
| [dusk-merkle](https://github.com/dusk-network/dusk-merkle) | Merkle tree implementation in Rust |
| [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) | Cryptographic hash function |
| [EZKL](https://github.com/zkonduit/ezkl) | Zero-knowledge proof system for neural networks |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | Cross-platform ML model inference |

<hr>

<p align="center">
<sub>Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International</sub>
</p>
