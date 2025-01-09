#!/usr/bin/env python3
"""
final_split_inference.py

A concise, minimal script for split inference with:
 - LoRA'd submodules in c_attn only,
 - combine_mode=add_delta,
 - verifying remote_out is smaller than base_out,
 - printing final cross-entropy vs. direct local LoRA.

Usage:
  python final_split_inference.py \
    --base_model distilgpt2 \
    --lora_model ng0-k1/distilgpt2-finetuned-es \
    --combine_mode add_delta \
    --text "Hello world, this is a LoRA test."
"""

import threading
import os
import sys
import time
import uuid
import pickle
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel

###################################
# Deterministic seed
###################################
SEED = 1234
def set_global_seed(s=1234):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_global_seed(SEED)

def strip_base_prefix(name: str):
    p = "base_model.model."
    if name.startswith(p):
        return name[len(p):]
    return name

###################################
# LoRAServer
###################################
class LoRAServer:
    def __init__(self, base_model, lora_model, debug=False):
        config = AutoConfig.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_config(config)
        base.eval()

        self.peft_model = PeftModel.from_pretrained(base, lora_model)
        self.peft_model.eval()

        # zero out any dropout
        for m in self.peft_model.modules():
            if hasattr(m, "lora_dropout"):
                m.lora_dropout.p = 0.0

        self.lora_subs = {}
        for raw_name, mod in self.peft_model.named_modules():
            if any("lora" in pn.lower() for pn, _ in mod.named_parameters()):
                fixed = strip_base_prefix(raw_name)
                self.lora_subs[fixed] = mod

        if debug:
            print("[LoRAServer Debug] submodules with LoRA:\n")
            for k in sorted(self.lora_subs.keys()):
                print(" ", k)

    def list_lora_injection_points(self):
        return list(self.lora_subs.keys())

    def apply_lora(self, sub_name, input_tensor: torch.Tensor):
        if sub_name not in self.lora_subs:
            raise ValueError(f"[Server] no LoRA submodule named '{sub_name}'")
        with torch.no_grad():
            out = self.lora_subs[sub_name](input_tensor)
        return out

###################################
# A-listener
###################################
class UserAListener(threading.Thread):
    def __init__(self, server, comm_folder="./temp_communication", stop_event=None):
        super().__init__()
        self.server = server
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)
        self.stop_event = stop_event or threading.Event()

    def run(self):
        print("[A-listener] Starting loop...")
        while not self.stop_event.is_set():
            reqs = glob.glob(os.path.join(self.comm_folder, "request_*.pkl"))
            for r in reqs:
                self.process_request(r)
            time.sleep(0.1)
        print("[A-listener] Stopped loop.")

    def process_request(self, pth):
        with open(pth,"rb") as f:
            data = pickle.load(f)
        rid = data["request_id"]
        rtype = data.get("request_type","lora_forward")

        if rtype=="init_request":
            submods = self.server.list_lora_injection_points()
            resp_data = {
                "request_id": rid,
                "response_type":"init_response",
                "injection_points": submods
            }
        else:
            subname = data["submodule_name"]
            arr = data["input_array"]
            inp = torch.tensor(arr)
            out_t = self.server.apply_lora(subname, inp)
            resp_data = {
                "request_id": rid,
                "response_type":"lora_forward_response",
                "output_array": out_t.numpy()
            }

        resp_path = os.path.join(self.comm_folder, f"response_{rid}.pkl")
        with open(resp_path,"wb") as f:
            pickle.dump(resp_data, f)
        os.remove(pth)

###################################
# B side
###################################
class UserBFileHandler:
    def __init__(self, folder="./temp_communication"):
        self.folder = folder
        os.makedirs(self.folder, exist_ok=True)

    def send_init_request(self):
        rid = str(uuid.uuid4())[:8]
        req_path = os.path.join(self.folder, f"request_{rid}.pkl")
        data = {"request_id":rid,"request_type":"init_request"}
        with open(req_path,"wb") as f:
            pickle.dump(data,f)

        resp_path = os.path.join(self.folder,f"response_{rid}.pkl")
        while not os.path.exists(resp_path):
            time.sleep(0.1)
        with open(resp_path,"rb") as f:
            resp_data = pickle.load(f)
        os.remove(resp_path)
        return resp_data["injection_points"]

    def send_lora_forward(self, sub_name, tensor: torch.Tensor):
        rid = str(uuid.uuid4())[:8]
        req_path = os.path.join(self.folder, f"request_{rid}.pkl")
        data = {
            "request_id":rid,
            "request_type":"lora_forward",
            "submodule_name":sub_name,
            "input_array":tensor.cpu().numpy()
        }
        with open(req_path,"wb") as f:
            pickle.dump(data,f)

        resp_path = os.path.join(self.folder,f"response_{rid}.pkl")
        while not os.path.exists(resp_path):
            time.sleep(0.1)
        with open(resp_path,"rb") as f:
            resp_data = pickle.load(f)
        os.remove(resp_path)

        arr = resp_data["output_array"]
        return torch.tensor(arr)

###################################
# RemoteLoRAWrappedModule
###################################
class RemoteLoRAWrappedModule(nn.Module):
    """
    We'll log base_out vs remote_out norm to confirm remote is a small delta.
    """
    def __init__(self, sub_name, local_submodule, b_file_handler, combine_mode="replace"):
        super().__init__()
        self.sub_name = sub_name
        self.local_submodule = local_submodule
        self.b_file_handler = b_file_handler
        self.combine_mode = combine_mode

    def forward(self, x):
        with torch.no_grad():
            base_out = self.local_submodule(x)
        remote_out = self.b_file_handler.send_lora_forward(self.sub_name, x)

        b_nr = base_out.norm().item()
        b_mx = base_out.abs().max().item()
        r_nr = remote_out.norm().item()
        r_mx = remote_out.abs().max().item()

        print(f"[Debug c_attn '{self.sub_name}'] "
              f"base_out norm={b_nr:.4f}, max={b_mx:.4f}; "
              f"remote_out norm={r_nr:.4f}, max={r_mx:.4f} => combine={self.combine_mode}")

        if self.combine_mode=="add_delta":
            return base_out + remote_out
        else:
            return remote_out

###################################
# BaseModelClient
###################################
class BaseModelClient:
    def __init__(self,
                 base_model,
                 file_handler,
                 combine_mode="replace"):
        set_global_seed(SEED)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        self.model.eval()

        self.file_handler = file_handler
        self.combine_mode = combine_mode

    def patch_submodules(self):
        all_subs = self.file_handler.send_init_request()
        print("[B] All submodules from A:", all_subs)

        # We'll only patch c_attn
        leftover = []
        to_patch = []
        for s in all_subs:
            if s.endswith(".c_attn"):
                to_patch.append(s)
            else:
                leftover.append(s)

        print("[B] We'll patch submodules:", to_patch)
        print("[B] leftover submodules (not patched):", leftover)

        for name in to_patch:
            try:
                parts = name.split(".")
                *parents, child = parts
                m = self.model
                for p in parents:
                    m = getattr(m, p)
                orig = getattr(m, child)
                wrapped = RemoteLoRAWrappedModule(
                    sub_name=name,
                    local_submodule=orig,
                    b_file_handler=self.file_handler,
                    combine_mode=self.combine_mode
                )
                setattr(m, child, wrapped)
                print(f"[B] Patched '{name}' -> RemoteLoRAWrappedModule.")
            except Exception as e:
                print(f"[B] Could not patch '{name}': {e}")

    def compute_loss(self, text:str):
        enc = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc, labels=enc["input_ids"])
        return out.loss.item()

###################################
# MAIN
###################################
def main():
    import threading

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="openai-community/gpt2")
    parser.add_argument("--lora_model", default="fzzhang/gpt2_GSM8K_lora_5e5")
    parser.add_argument("--combine_mode", default="add_delta", choices=["replace","add_delta"])
    parser.add_argument("--text", default="Hello world, this is a LoRA test.")
    args = parser.parse_args()

    COMM="./temp_communication"
    os.makedirs(COMM, exist_ok=True)

    # Start A
    server = LoRAServer(args.base_model, args.lora_model, debug=True)
    stop_evt = threading.Event()
    listener = UserAListener(server, comm_folder=COMM, stop_event=stop_evt)
    listener.start()

    # B side
    b_handler = UserBFileHandler(COMM)
    base_client = BaseModelClient(
        base_model=args.base_model,
        file_handler=b_handler,
        combine_mode=args.combine_mode
    )
    base_client.patch_submodules()

    # 1) Split inference
    split_loss = base_client.compute_loss(args.text)
    print(f"\n[Split normal forward] loss={split_loss:.4f}")

    # stop
    stop_evt.set()
    listener.join()
    print("[A-listener] Stopped listening loop.")

    # 2) Direct LoRA
    direct_base = AutoModelForCausalLM.from_pretrained(args.base_model)
    direct_base.eval()
    direct_lora = PeftModel.from_pretrained(direct_base, args.lora_model)
    direct_lora.eval()

    # zero out dropout
    for m in direct_lora.modules():
        if hasattr(m,"lora_dropout"):
            m.lora_dropout.p = 0.0

    tok = base_client.tokenizer(args.text, return_tensors="pt")
    with torch.no_grad():
        out_dir = direct_lora(**tok, labels=tok["input_ids"])
        direct_loss = out_dir.loss.item()
    print(f"[Direct LoRA] loss={direct_loss:.4f}")

    print("\n[Done final minimal script]")


if __name__=="__main__":
    main()
