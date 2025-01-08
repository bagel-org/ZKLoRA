#!/usr/bin/env python3
"""
File: b_client.py

User B's code:
  - Loads the base distilgpt2 model (no LoRA).
  - Asks User A (via file-based requests) for the LoRA injection points.
  - Monkey-patches the relevant submodules to call A for LoRA transformations.
  - Then does a forward pass to compute token-level loss on a sample text.

Usage:
  python b_client.py
"""

import os
import time
import uuid
import pickle
import argparse

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class UserBFileHandler:
    """
    Creates request_*.pkl in a shared folder, waits for response_*.pkl from A.
    """

    def __init__(self, comm_folder="./temp_communication"):
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)

    def send_init_request(self) -> list:
        request_id = str(uuid.uuid4())[:8]
        request_path = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        data = {
            "request_id": request_id,
            "request_type": "init_request",
        }
        with open(request_path, "wb") as f:
            pickle.dump(data, f)

        # Wait for response
        response_path = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while not os.path.exists(response_path):
            time.sleep(0.1)

        with open(response_path, "rb") as f:
            resp_data = pickle.load(f)
        os.remove(response_path)

        return resp_data["injection_points"]

    def send_lora_request_and_wait(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        request_id = str(uuid.uuid4())[:8]
        req_path = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        data = {
            "request_id": request_id,
            "request_type": "lora_forward",
            "submodule_name": submodule_name,
            "input_array": input_tensor.cpu().numpy(),
        }
        with open(req_path, "wb") as f:
            pickle.dump(data, f)

        # Wait for response
        resp_path = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while not os.path.exists(resp_path):
            time.sleep(0.1)

        with open(resp_path, "rb") as f:
            resp_data = pickle.load(f)
        os.remove(resp_path)

        out_array = resp_data["output_array"]
        return torch.tensor(out_array, dtype=torch.float32)


class RemoteLoRAWrappedModule(nn.Module):
    """
    Replaces local submodule with a remote LoRA call to A.
    """

    def __init__(self, submodule_name: str, local_submodule: nn.Module,
                 b_file_handler: UserBFileHandler, combine_mode="replace"):
        super().__init__()
        self.submodule_name = submodule_name
        self.local_submodule = local_submodule
        self.b_file_handler = b_file_handler
        self.combine_mode = combine_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            base_out = self.local_submodule(x)

        lora_out = self.b_file_handler.send_lora_request_and_wait(self.submodule_name, x)

        if self.combine_mode == "add_delta":
            return base_out + lora_out
        else:
            return lora_out


class BaseModelClient:
    """
    B loads base distilgpt2 (no LoRA), obtains injection points from A,
    then monkey-patches them. 
    Finally, can compute token-level loss on a text.
    """
    def __init__(self, base_model_name="distilgpt2",
                 b_file_handler=None,
                 combine_mode="replace"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.model.eval()

        self.b_file_handler = b_file_handler
        self.combine_mode = combine_mode

    def discover_and_patch_lora(self):
        # 1) Ask A for injection points
        injection_points = self.b_file_handler.send_init_request()
        print("[B] Injection points received:", injection_points)

        # 2) Filter for 'c_attn' if we only want to patch those
        submodules_to_patch = [x for x in injection_points if "attn.c_attn" in x and x]

        # 3) Monkey-patch each
        for name in submodules_to_patch:
            try:
                parts = name.split(".")
                *parents, child = parts
                m = self.model
                for p in parents:
                    m = getattr(m, p)
                original_submod = getattr(m, child)

                wrapped = RemoteLoRAWrappedModule(
                    submodule_name=name,
                    local_submodule=original_submod,
                    b_file_handler=self.b_file_handler,
                    combine_mode=self.combine_mode,
                )
                setattr(m, child, wrapped)
                print(f"[B] Patched '{name}' => RemoteLoRAWrappedModule.")
            except Exception as e:
                print(f"[B] Could not patch '{name}': {e}")

    def compute_token_loss(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = self.model(input_ids, labels=input_ids)
        return out.loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="distilgpt2")
    parser.add_argument("--comm_folder", type=str, default="./temp_communication")
    parser.add_argument("--combine_mode", type=str, default="replace",
                        help="Use 'replace' if remote outputs (base+LoRA). Use 'add_delta' if remote is only LoRA delta.")
    args = parser.parse_args()

    file_handler = UserBFileHandler(comm_folder=args.comm_folder)
    base_client = BaseModelClient(
        base_model_name=args.base_model,
        b_file_handler=file_handler,
        combine_mode=args.combine_mode
    )

    # 1) Ask A for injection points + patch
    base_client.discover_and_patch_lora()

    # 2) Compute token-level loss
    text = "Hello world, this is a LoRA test."
    loss_val = base_client.compute_token_loss(text)
    print(f"[B] Computed token-level loss: {loss_val:.4f}")


if __name__ == "__main__":
    main()
