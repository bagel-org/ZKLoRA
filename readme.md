## ZKLoRA: Efficient Zero-Knowledge Proofs for LoRA Verification

Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), have revolutionized how large-scale language models are specialized for new tasks. These approaches reduce the computational and memory overhead drastically compared to full fine-tuning, making them highly efficient and practical. However, real-world deployment often faces a **trust dilemma**:

1. **Verification by the Base Model User**: The base model user would like to leverage a set of LoRA paramters on top of a base model, but needs verification that those LoRA parameters perform well on the target task.
2. **Protection for the LoRA Contributor**: The contributor invests resources in fine-tuning and needs assurance of fair compensation without prematurely revealing the LoRA parameters.

**ZKLoRA** addresses this dilemma using **zero-knowledge proofs** to securely verify LoRA updates without exposing the parameters themselves. Our approach:

- Enables **secure proof** that a LoRA update was derived from a specific base model.
- Uses **polynomial commitments** and **succinct ZK proofs** to allow near real-time verification, even for large models.

### ZKLoRA + Multi-Party Inference (MPI)

We apply ZKLoRA to a **multi-party inference** scenario, where:
- **User A** (LoRA contributor) holds LoRA-augmented submodules.
- **User B** (base model user) has the large base model.
- They **collaborate** on inference while ensuring the LoRA computations remain hidden, and **A** can generate zero-knowledge proofs of the LoRA inference correctness.

**After** the inference run, **A** creates proof artifacts offline (e.g., `.onnx`, `.json`, plus the proof files) and **B** can download them to locally verify that the LoRA computations were performed faithfully.

---

## Quick Usage Instructions

Below are **two** sample scripts illustrating multi-party inference with proof generation. **A** is the LoRA contributor, **B** is the base model user.

### 1. LoRA Provider Side (User A)

Use `lora_contributor_sample_script.py` (or an equivalent script in your project) to:

1. **Host** the LoRA submodules on a specific IP & port.
2. Listen for requests from **B** during inference (e.g., forward pass submodules).
3. Collect the submodule inputs, generate proof artifacts once inference ends, and store them locally in an output folder.

A minimal sample script might look like:

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

When **B** calls “end_inference,” this script synchronously finalizes proof generation and stores proof artifacts (e.g., in `a-out/`). **A** can then share or host these files for **B** to download and verify offline.

### 2. Base Model User Side (User B)

Use `base_model_user_sample_script.py` to:

1. Load the original base model from `from_pretrained`.
2. Connect to **A**’s submodules (via IP & port).
3. “Patch” the relevant submodules with remote LoRA calls.
4. Perform a forward pass to get token-level loss (or any needed inference).
5. Call “end_inference” so **A** can generate proofs locally. 
6. **B** then retrieves the proof files from **A** by an out-of-band transfer (e.g., scp or HTTP).

A minimal script:

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

**That’s it**—once B calls `client.end_inference()`, **A** does the proof generation. B obtains those proof files from the `a-out/` folder by some external means (scp, HTTP, etc.) and can verify them locally.

---

## Verifying Proof Files Locally

Once **B** has accessed the proof artifacts from **A** (for example, the files in `a-out/`), the next step is to run a **verification script** that calls `batch_verify_proofs`. Here’s a sample `verify_proofs.py`:

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

This script uses `batch_verify_proofs(...)` to:

1. **Scan** for `.pf` proof files in `--proof_dir`.
2. For each proof, **resolve** the associated settings, verification key, and SRS paths.
3. Call the underlying `ezkl.verify(...)` method to confirm each proof is valid.
4. Print out how many proofs were verified and how long the process took.

You can set `--verbose` for extra logs about verification timing and success/failure messages.

---

## Summary

- **ZKLoRA** solves a trust problem in LoRA fine-tuning: verifying that LoRA updates or inferences are derived from a legitimate base model, without exposing the LoRA parameters.
- **Multi-Party Inference** workflow lets a base model user (B) and a LoRA contributor (A) do secure inference, with **A** generating zero-knowledge proofs offline.
- **A** runs `lora_contributor_sample_script.py`. **B** runs `base_model_user_sample_script.py`. 
- After inference, **B** obtains `a-out/` proof files from **A** and verifies them with `verify_proofs.py`.

ZKLoRA opens up new avenues for **secure, trust-minimized** AI collaboration using **PEFT** methods—enabling verifiable LoRA updates **and** multi-party inference with minimal overhead.