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

```