#!/usr/bin/env python3
"""
File: a_server.py

User A's code:
  - Loads a PEFT LoRAServer (distilgpt2 + LoRA adapter).
  - Runs a 'UserAListener' thread that continuously scans for request_*.pkl
    in a shared folder and writes response_*.pkl with results.

Usage Example:
  python a_server.py --base_model distilgpt2 --lora_model_id my-lora-distilgpt2
"""

import os
import time
import uuid
import pickle
import glob
import threading
import argparse

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel


def strip_base_prefix(name: str) -> str:
    """
    If submodule name starts with 'base_model.model.', remove that prefix
    so it aligns with the raw GPT2 submodule naming.
    """
    prefix = "base_model.model."
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


class LoRAServer:
    """
    Loads a base model + LoRA adapter via PEFT, and stores submodule references
    keyed by the *stripped* name (e.g., 'transformer.h.0.attn.c_attn').
    """

    def __init__(self, base_model_name, lora_model_id):
        # 1) Load base config/model
        config = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_config(config)
        base_model.eval()

        # 2) Load LoRA adapter
        self.peft_model = PeftModel.from_pretrained(base_model, lora_model_id)
        self.peft_model.eval()

        # 3) Build submodule dict
        self.lora_submodules = {}
        for raw_name, module in self.peft_model.named_modules():
            if any("lora" in pname.lower() for pname, _ in module.named_parameters()):
                fixed_name = strip_base_prefix(raw_name)
                self.lora_submodules[fixed_name] = module

    def list_lora_injection_points(self):
        return list(self.lora_submodules.keys())

    def apply_lora(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        if submodule_name not in self.lora_submodules:
            raise ValueError(f"No LoRA submodule named '{submodule_name}'")
        with torch.no_grad():
            out = self.lora_submodules[submodule_name](input_tensor)
        return out


class UserAListener(threading.Thread):
    """
    Background thread that polls for request_*.pkl in comm_folder,
    processes them with LoRAServer, and writes response_*.pkl.
    """

    def __init__(self, lora_server: LoRAServer, comm_folder: str, stop_event=None):
        super().__init__()
        self.lora_server = lora_server
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)
        self.stop_event = stop_event or threading.Event()

    def run(self):
        print("[A-listener] Starting listening loop...")
        while not self.stop_event.is_set():
            # Find request files
            requests = glob.glob(os.path.join(self.comm_folder, "request_*.pkl"))
            for req_file in requests:
                self.process_request_file(req_file)
            time.sleep(0.2)
        print("[A-listener] Stopped listening loop.")

    def process_request_file(self, request_path: str):
        with open(request_path, "rb") as f:
            request_data = pickle.load(f)

        req_id = request_data["request_id"]
        req_type = request_data.get("request_type", "lora_forward")

        if req_type == "init_request":
            # Return injection points
            points = self.lora_server.list_lora_injection_points()
            response_data = {
                "request_id": req_id,
                "response_type": "init_response",
                "injection_points": points,
            }
        else:
            # LoRA forward
            submodule_name = request_data["submodule_name"]
            input_array = request_data["input_array"]
            input_tensor = torch.tensor(input_array, dtype=torch.float32)

            output_tensor = self.lora_server.apply_lora(submodule_name, input_tensor)
            response_data = {
                "request_id": req_id,
                "response_type": "lora_forward_response",
                "output_array": output_tensor.cpu().numpy(),
            }

        response_file = os.path.join(self.comm_folder, f"response_{req_id}.pkl")
        with open(response_file, "wb") as f:
            pickle.dump(response_data, f)

        # Remove the request so we don't reprocess it
        os.remove(request_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="distilgpt2")
    parser.add_argument("--lora_model_id", type=str, default="my-lora-distilgpt2")
    parser.add_argument("--comm_folder", type=str, default="./temp_communication")
    args = parser.parse_args()

    # Create the LoRAServer
    lora_server = LoRAServer(args.base_model, args.lora_model_id)

    # Start listener thread
    stop_event = threading.Event()
    listener = UserAListener(lora_server, comm_folder=args.comm_folder, stop_event=stop_event)
    listener.start()

    print("[A-server] Running. Press Ctrl+C to stop or kill process.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[A-server] Shutting down...")

    stop_event.set()
    listener.join()


if __name__ == "__main__":
    main()
