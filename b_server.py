#!/usr/bin/env python3
"""
User B: 
 - Receives submodules from A (like 'transformer.h.0.attn.c_attn' etc.).
 - Patches them => forward => triggers A's submodules => .onnx => synchronous proof gen in A => returns proofs => B reads them
 - We do config.use_cache=False so no 'past_key_values'.
 - If you pass --verify_proofs, we do local batch_verify_proofs afterwards.
"""

import argparse
import socket
import pickle
import time
import os

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft.tuners.lora.layer import LoraLayer

from zklora import batch_verify_proofs

class BToAComm:
    def __init__(self, host_a="127.0.0.1", port_a=30000):
        self.host_a = host_a
        self.port_a = port_a

    def init_request(self):
        data = {"request_type": "init_request"}
        resp = self.send_and_recv(data)
        return resp.get("injection_points", [])

    def lora_forward(self, sub_name, arr):
        req = {
            "request_type":"lora_forward",
            "submodule_name": sub_name,
            "input_array": arr
        }
        resp = self.send_and_recv(req)
        return resp.get("output_array", None)

    def end_inference(self):
        req = {"request_type":"end_inference"}
        resp = self.send_and_recv(req)
        return resp.get("proof_map", {})

    def send_and_recv(self, data_dict):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host_a, self.port_a))
        bin_req = pickle.dumps(data_dict)
        s.sendall(bin_req)
        s.shutdown(socket.SHUT_WR)

        buffer = b""
        s.settimeout(2400.0)  # give more time if proof generation is slow
        while True:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buffer += chunk
        s.close()

        if not buffer:
            raise RuntimeError("[B] No data from A (EOF). Possibly A took too long or closed early.")

        resp = pickle.loads(buffer)
        return resp

class RemoteLoRAWrappedModule(nn.Module):
    def __init__(self, sub_name, local_sub, comm: BToAComm, combine_mode="replace"):
        super().__init__()
        self.sub_name = sub_name
        self.local_sub = local_sub
        self.comm = comm
        self.combine_mode = combine_mode

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            base_out = self.local_sub(x)
        arr = x.cpu().numpy()
        remote_out = self.comm.lora_forward(self.sub_name, arr)
        if remote_out is None:
            raise RuntimeError(f"[B] submodule '{self.sub_name}' => no output from A.")
        out_t = torch.tensor(remote_out, dtype=torch.float32)
        if self.combine_mode == "add_delta":
            return base_out + out_t
        return out_t

class BaseModelClient:
    def __init__(self, base_model="distilgpt2", host_a="127.0.0.1", port_a=30000, combine_mode="replace"):
        # turn off cache => no 'past_key_values'
        #config = AutoConfig.from_pretrained(base_model)
        #config.use_cache = False
        self.model = AutoModelForCausalLM.from_pretrained(base_model)
        
        self.model.config.use_cache = False
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        self.comm = BToAComm(host_a, port_a)
        self.combine_mode = combine_mode

    def _navigate(self, mod: nn.Module, parts: list[str]) -> nn.Module:
        """
        If a part is digits => mod=mod[int], else mod=getattr(mod, part).
        E.g. 'transformer','h','0','attn','c_attn' => indexing for '0'.
        """
        for p in parts:
            if p.isdigit():
                idx = int(p)
                mod = mod[idx]
            else:
                mod = getattr(mod, p)
        return mod

    def init_and_patch(self):
        submods = self.comm.init_request()
        print("[B] injection points =>", submods)
        for full_name in submods:
            if not full_name.strip():
                print("[B] skipping empty submodule name.")
                continue
            try:
                path_parts = full_name.split(".")
                *parents, child = path_parts
                m = self._navigate(self.model, parents)
                orig_sub = getattr(m, child)
                wrapped = RemoteLoRAWrappedModule(full_name, orig_sub, self.comm, self.combine_mode)
                setattr(m, child, wrapped)
                print(f"[B] Patched submodule '{full_name}'.")
            except Exception as e:
                print(f"[B] Could not patch '{full_name}': {e}")

    def forward_loss(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt")
        in_ids = enc["input_ids"]
        with torch.no_grad():
            out = self.model(in_ids, labels=in_ids)
        return out.loss.item()

    def end_inference_and_retrieve_proofs(self):
        proof_map = self.comm.end_inference()
        out_dir = "b-out"
        os.makedirs(out_dir, exist_ok=True)

        for base_name, fdict in proof_map.items():
            pf = os.path.join(out_dir, f"{base_name}.pf")
            with open(pf, "wb") as fp:
                fp.write(fdict["proof"])

            st = os.path.join(out_dir, f"{base_name}_settings.json")
            with open(st, "wb") as fp:
                fp.write(fdict["settings"])

            vk = os.path.join(out_dir, f"{base_name}.vk")
            with open(vk, "wb") as fp:
                fp.write(fdict["verification_key"])

            srs = os.path.join(out_dir, "kzg.srs")
            with open(srs, "wb") as fp:
                fp.write(fdict["srs"])

            print(f"[B] wrote proof artifacts for '{base_name}' => {out_dir}/")

        return out_dir

def debug_hook_factory(module_name):
    """
    Returns a hook function capturing the shape of the main input tensor.
    Adjust as needed for more debug info (e.g. printing outputs, stats).
    """
    def debug_hook(module, inputs, outputs):
        # inputs is a tuple of Tensors; typically inputs[0] is the main hidden state
        if len(inputs) == 0:
            # sometimes might be empty if the module gets no direct input
            print(f"[DEBUG] {module_name} got NO inputs?!")
            return

        x = inputs[0]
        print(f"[DEBUG] Hook at '{module_name}': input shape={tuple(x.shape)}")
    return debug_hook

def register_hooks_until_lora(model):
    first_lora_encountered = False

    for module_name, module_obj in model.named_modules():
        # If we've already found a LoRA layer, skip hooking further modules.
        if first_lora_encountered:
            break

        # Check if this module is a LoRA layer
        if isinstance(module_obj, RemoteLoRAWrappedModule):
            print(f"[DEBUG] Found first LoRA layer at '{module_name}'. Stopping hook registration.")
            first_lora_encountered = True
            # We do NOT register a hook on the LoRA layer or anything deeper
            continue

        # Otherwise, register a debug hook
        hook_fn = debug_hook_factory(module_name)
        module_obj.register_forward_hook(hook_fn)

def debug_hook(module, inputs, outputs):
    """
    A hook to inspect the shapes/values going into (and optionally out of) `module`.
    `inputs` is a tuple of Tensors input to the module.
    `outputs` is the module's output tensor(s).
    """
    # Typically, `inputs[0]` is the main hidden-state Tensor passed to c_attn.
    if len(inputs) > 0:
        x = inputs[0]
        print(f"[DEBUG-HOOK: c_attn] Input shape: {tuple(x.shape)}")
        # If you want to see a snippet of values:
        #print(f"[DEBUG-HOOK: c_attn] Input (first row, first 5 dims): {x[0, 0, :5]}")
        print(f"[DEBUG-HOOK: c_attn] Input (first row, first 5 dims): {x[0, :5]}")
    else:
        print("[DEBUG-HOOK: c_attn] No inputs??")

    # If you'd like to see the module’s output:
    if isinstance(outputs, torch.Tensor):
        print(f"[DEBUG-HOOK: c_attn] Output shape: {tuple(outputs.shape)}")
        print(f"[DEBUG-HOOK: c_attn] outputs (first row, first 5 dims): {outputs[0, 0, :5]}")
    else:
        # Might be a tuple for some modules
        print(f"[DEBUG-HOOK: c_attn] Output is not a single Tensor: type={type(outputs)}")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_a", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument("--combine_mode", choices=["replace","add_delta"], default="add_delta")
    parser.add_argument("--verify_proofs", action="store_true")
    args = parser.parse_args()

    client = BaseModelClient(args.base_model, args.host_a, args.port_a, args.combine_mode)
    client.init_and_patch()

    # forward pass => triggers submodule calls => A accumulates => .onnx => proofs
    text = "Hello World, this is a LoRA test."

    loss_val = client.forward_loss(text)
    print(f"[B] final loss => {loss_val:.4f}")

    out_dir = client.end_inference_and_retrieve_proofs()
    print("[B] local proofs =>", out_dir)

    if args.verify_proofs:
        print("[B] verifying proofs in", out_dir)
        total_t, c = batch_verify_proofs(proof_dir=out_dir, verbose=True)
        print(f"[B] verified {c} proof(s) in {total_t:.2f} sec.")

if __name__=="__main__":
    main()
