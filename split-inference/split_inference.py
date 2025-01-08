import os
import time
import uuid
import pickle
import glob
import threading

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


###################################
# 1) Utility function for name-fixing
###################################
def strip_base_prefix(name: str) -> str:
    """
    If submodule name starts with 'base_model.model.', remove that prefix
    so that it matches the underlying GPT2 submodule naming.
    Example:
      'base_model.model.transformer.h.0.attn.c_attn'
      -> 'transformer.h.0.attn.c_attn'
    """
    prefix = "base_model.model."
    if name.startswith(prefix):
        return name[len(prefix):]  # remove the prefix
    return name


###################################
# 2) LoRAServer (User A)
###################################
class LoRAServer:
    """
    Holds a base model + LoRA adapter from PEFT and can apply LoRA submodules on request.
    """

    def __init__(self, base_model_name="distilgpt2", lora_model_id="my-lora-distilgpt2"):
        # Load base config and model
        self.config = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_config(self.config)
        base_model.eval()

        # Load LoRA
        self.peft_model = PeftModel.from_pretrained(base_model, lora_model_id)
        self.peft_model.eval()

        # Build a dict of LoRA submodules
        # BUT store them with 'base_model.model.' stripped,
        # so we effectively have keys like 'transformer.h.0.attn.c_attn'.
        self.lora_submodules = {}
        for raw_name, module in self.peft_model.named_modules():
            # If we see a param containing 'lora', assume this submodule has LoRA
            if any("lora" in pname.lower() for pname, _ in module.named_parameters()):
                fixed_name = strip_base_prefix(raw_name)
                self.lora_submodules[fixed_name] = module

    def list_lora_injection_points(self):
        """
        Return the submodule keys that we've stored (after prefix-stripping).
        """
        return list(self.lora_submodules.keys())

    def apply_lora(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform (base + LoRA) for the requested submodule.
        """
        if submodule_name not in self.lora_submodules:
            raise ValueError(f"No LoRA submodule named '{submodule_name}'")
        with torch.no_grad():
            out = self.lora_submodules[submodule_name](input_tensor)
        return out


###################################
# 3) UserAListener (Thread)
###################################
class UserAListener(threading.Thread):
    """
    A separate thread that polls for request_*.pkl. On receiving:
      - 'init_request': returns the submodule names from LoRAServer
      - 'lora_forward': applies LoRA submodule to the input activations
    """
    def __init__(self, lora_server: LoRAServer, comm_folder="./temp_communication", stop_event=None):
        super().__init__()
        self.lora_server = lora_server
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)
        self.stop_event = stop_event or threading.Event()

    def run(self):
        print("[A-listener] Starting listening loop...")
        while not self.stop_event.is_set():
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
            # Return submodule injection points
            injection_points = self.lora_server.list_lora_injection_points()
            response_data = {
                "request_id": req_id,
                "response_type": "init_response",
                "injection_points": injection_points,
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

        os.remove(request_path)


###################################
# 4) UserBFileHandler
###################################
class UserBFileHandler:
    """
    On B's side, we write requests and wait for responses.
    """
    def __init__(self, comm_folder="./temp_communication"):
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)

    def send_init_request(self) -> list:
        request_id = str(uuid.uuid4())[:8]
        req_file = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        request_data = {
            "request_id": request_id,
            "request_type": "init_request",
        }
        with open(req_file, "wb") as f:
            pickle.dump(request_data, f)

        response_file = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while not os.path.exists(response_file):
            time.sleep(0.1)

        with open(response_file, "rb") as f:
            response_data = pickle.load(f)
        os.remove(response_file)
        return response_data["injection_points"]

    def send_lora_request(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        request_id = str(uuid.uuid4())[:8]
        req_file = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        request_data = {
            "request_id": request_id,
            "request_type": "lora_forward",
            "submodule_name": submodule_name,
            "input_array": input_tensor.cpu().numpy(),
        }
        with open(req_file, "wb") as f:
            pickle.dump(request_data, f)

        response_file = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while not os.path.exists(response_file):
            time.sleep(0.1)

        with open(response_file, "rb") as f:
            response_data = pickle.load(f)
        os.remove(response_file)

        out_array = response_data["output_array"]
        return torch.tensor(out_array, dtype=torch.float32)


###################################
# 5) RemoteLoRAWrappedModule
###################################
class RemoteLoRAWrappedModule(nn.Module):
    """
    Replaces local submodule with a remote LoRA submodule call.
    """
    def __init__(self, submodule_name: str, local_submodule: nn.Module, file_handler_B: UserBFileHandler, combine_mode="replace"):
        super().__init__()
        self.submodule_name = submodule_name
        self.local_submodule = local_submodule
        self.file_handler_B = file_handler_B
        self.combine_mode = combine_mode

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # run the base model's submodule
        with torch.no_grad():
            base_out = self.local_submodule(input_tensor)

        # remote call for LoRA
        lora_out = self.file_handler_B.send_lora_request(self.submodule_name, input_tensor)

        # combine
        if self.combine_mode == "add_delta":
            return base_out + lora_out
        else:
            return lora_out


###################################
# 6) BaseModelClient (User B)
###################################
class BaseModelClient:
    """
    Loads the base model from HF, gets the LoRA injection points from A,
    then monkey-patches them, and can compute token-level loss.
    """

    def __init__(self, base_model_name="distilgpt2", file_handler_B=None, combine_mode="replace"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_model.eval()

        self.file_handler_B = file_handler_B
        self.combine_mode = combine_mode

    def discover_and_patch_lora(self):
        """
        1) Get the LoRA injection points from A
        2) Patch each relevant submodule, ignoring blank or extraneous ones
        """
        injection_points = self.file_handler_B.send_init_request()
        print("[B] Received LoRA injection points from A:", injection_points)

        # Filter for 'c_attn' just as an example
        c_attn_subs = [ip for ip in injection_points if "attn.c_attn" in ip and ip != ""]

        # Monkey-patch them
        for name in c_attn_subs:
            try:
                # The server gave us something like "transformer.h.0.attn.c_attn"
                # We need to ensure that the local base model has that path.
                # It's already stripped of 'base_model.model.' on A's side.
                parts = name.split(".")
                *parent_parts, child_name = parts
                m = self.base_model
                for p in parent_parts:
                    m = getattr(m, p)
                original = getattr(m, child_name)

                wrapped = RemoteLoRAWrappedModule(
                    submodule_name=name,
                    local_submodule=original,
                    file_handler_B=self.file_handler_B,
                    combine_mode=self.combine_mode
                )
                setattr(m, child_name, wrapped)

                print(f"[B] Patched '{name}' as RemoteLoRAWrappedModule.")
            except Exception as e:
                print(f"[B] Could not patch '{name}': {e}")

    def compute_token_loss(self, text: str):
        """
        Just compute cross-entropy token-level loss
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            out = self.base_model(input_ids=inputs["input_ids"], labels=inputs["input_ids"])
        return out.loss.item()


###################################
# 7) MAIN
###################################
def main():
    comm_folder = "./temp_communication"
    os.makedirs(comm_folder, exist_ok=True)

    # A) Start LoRAServer + thread
    lora_server = LoRAServer("distilgpt2", "ng0-k1/distilgpt2-finetuned-es")
    stop_event = threading.Event()
    a_listener = UserAListener(lora_server, comm_folder=comm_folder, stop_event=stop_event)
    a_listener.start()

    # B) B sets up
    b_file_handler = UserBFileHandler(comm_folder=comm_folder)
    base_client = BaseModelClient(
        base_model_name="distilgpt2",
        file_handler_B=b_file_handler,
        combine_mode="replace"
    )

    # B1) B discovers injection points and patches them
    base_client.discover_and_patch_lora()

    # B2) Test compute token-level loss
    sample_text = "Hello world, this is a LoRA test."
    loss_val = base_client.compute_token_loss(sample_text)
    print(f"[B] Computed token-level loss: {loss_val:.4f}")

    # Cleanup
    stop_event.set()
    a_listener.join()
    print("[Main] Done.")


if __name__ == "__main__":
    main()
