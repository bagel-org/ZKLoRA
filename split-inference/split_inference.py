import os
import time
import uuid
import pickle
import glob
import threading

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


###############################
# 1) LoRAServer (User A)
###############################
class LoRAServer:
    """
    Holds a base model + LoRA adapter (loaded from a pretrained checkpoint)
    and can apply LoRA submodules on request.
    """

    def __init__(self, base_model_name="distilgpt2", lora_model_id="my-lora-distilgpt2"):
        self.config = AutoConfig.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_config(self.config)
        base_model.eval()

        # Load LoRA-adapted model from PEFT
        self.peft_model = PeftModel.from_pretrained(base_model, lora_model_id)
        self.peft_model.eval()

        # Collect submodule references that have LoRA
        self.lora_submodules = {}
        for name, module in self.peft_model.named_modules():
            # naive check: if any param name has 'lora'
            if any("lora" in pname.lower() for pname, _ in module.named_parameters()):
                self.lora_submodules[name] = module

    def list_lora_injection_points(self):
        return list(self.lora_submodules.keys())

    def apply_lora(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        if submodule_name not in self.lora_submodules:
            raise ValueError(f"No LoRA submodule named '{submodule_name}'")
        with torch.no_grad():
            out = self.lora_submodules[submodule_name](input_tensor)
        return out


###############################
# 2) UserAListener (Thread)
###############################
class UserAListener(threading.Thread):
    """
    Runs in a background thread, continuously scanning for request_*.pkl files.
    When found, processes them via LoRAServer and writes response_*.pkl.

    We also handle a special "init_request" to provide injection points.
    """

    def __init__(self, 
                 lora_server: LoRAServer, 
                 comm_folder="./temp_communication", 
                 stop_event=None):
        super().__init__()
        self.lora_server = lora_server
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)
        self.stop_event = stop_event or threading.Event()

    def run(self):
        print("[A-listener] Starting listening loop...")
        while not self.stop_event.is_set():
            # Find any request_*.pkl files
            requests = glob.glob(os.path.join(self.comm_folder, "request_*.pkl"))
            for req_file in requests:
                self.process_request_file(req_file)
            
            # Sleep briefly before checking again
            time.sleep(0.2)
        print("[A-listener] Stopped listening loop.")

    def process_request_file(self, request_path: str):
        # Read the request
        with open(request_path, "rb") as f:
            request_data = pickle.load(f)

        request_id = request_data["request_id"]
        request_type = request_data.get("request_type", "lora_forward")

        if request_type == "init_request":
            # B wants a list of injection points
            injection_points = self.lora_server.list_lora_injection_points()

            response_data = {
                "request_id": request_id,
                "response_type": "init_response",
                "injection_points": injection_points
            }
            response_filename = f"response_{request_id}.pkl"
            response_path = os.path.join(self.comm_folder, response_filename)
            with open(response_path, "wb") as f:
                pickle.dump(response_data, f)

            os.remove(request_path)

        else:
            # Normal LoRA forward request
            submodule_name = request_data["submodule_name"]
            input_array = request_data["input_array"]
            input_tensor = torch.tensor(input_array, dtype=torch.float32)

            # Apply LoRA
            output_tensor = self.lora_server.apply_lora(submodule_name, input_tensor)
            out_array = output_tensor.cpu().numpy()

            # Write response
            response_data = {
                "request_id": request_id,
                "response_type": "lora_forward_response",
                "output_array": out_array
            }
            response_filename = f"response_{request_id}.pkl"
            response_path = os.path.join(self.comm_folder, response_filename)
            with open(response_path, "wb") as f:
                pickle.dump(response_data, f)

            # Remove request file so we don't re-process it
            os.remove(request_path)


###############################
# 3) UserBFileHandler
###############################
class UserBFileHandler:
    """
    On B's side, we create a request file for each call and wait for the response.
    """

    def __init__(self, comm_folder="./temp_communication"):
        self.comm_folder = comm_folder
        os.makedirs(self.comm_folder, exist_ok=True)

    def send_init_request(self) -> list:
        """
        Asks A for the LoRA injection points. Returns a list of injection points.
        """
        request_id = str(uuid.uuid4())[:8]
        req_file = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        request_data = {
            "request_id": request_id,
            "request_type": "init_request",
        }
        with open(req_file, "wb") as f:
            pickle.dump(request_data, f)

        # Now wait for init_response
        response_file = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while True:
            if os.path.exists(response_file):
                with open(response_file, "rb") as f:
                    response_data = pickle.load(f)
                os.remove(response_file)
                return response_data["injection_points"]
            time.sleep(0.1)

    def send_lora_request_and_wait_for_response(self, submodule_name: str, input_tensor: torch.Tensor) -> torch.Tensor:
        # 1) Create a unique request ID
        request_id = str(uuid.uuid4())[:8]

        # 2) Save request file
        req_file = os.path.join(self.comm_folder, f"request_{request_id}.pkl")
        request_data = {
            "request_id": request_id,
            "request_type": "lora_forward",
            "submodule_name": submodule_name,
            "input_array": input_tensor.cpu().numpy()
        }
        with open(req_file, "wb") as f:
            pickle.dump(request_data, f)

        # 3) Wait (poll) for the response file
        response_file = os.path.join(self.comm_folder, f"response_{request_id}.pkl")
        while True:
            if os.path.exists(response_file):
                with open(response_file, "rb") as f:
                    response_data = pickle.load(f)
                os.remove(response_file)
                out_array = response_data["output_array"]
                return torch.tensor(out_array, dtype=torch.float32)
            time.sleep(0.1)


###############################
# 4) RemoteLoRAWrappedModule
###############################
class RemoteLoRAWrappedModule(nn.Module):
    """
    Replaces a local submodule with a 'remote' LoRA call (plus local base).
    """

    def __init__(self,
                 submodule_name: str,
                 local_submodule: nn.Module,
                 file_handler_B: UserBFileHandler,
                 combine_mode="replace"):
        super().__init__()
        self.submodule_name = submodule_name
        self.local_submodule = local_submodule
        self.file_handler_B = file_handler_B
        self.combine_mode = combine_mode

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # 1) local base
        with torch.no_grad():
            base_out = self.local_submodule(input_tensor)

        # 2) remote LoRA
        lora_out = self.file_handler_B.send_lora_request_and_wait_for_response(
            self.submodule_name, input_tensor
        )

        # 3) combine
        if self.combine_mode == "add_delta":
            return base_out + lora_out
        else:
            return lora_out


###############################
# -- NEW HELPER: Fix submodule names
###############################
def fix_peft_submodule_name(peft_name: str) -> str:
    """
    PEFT sometimes prefixes submodule names with 'base_model.model.' 
    but GPT2/DistilGPT2's actual submodule path is 'transformer.h.X.attn.c_attn', etc.
    
    This helper strips 'base_model.model.' if present, returning the 
    correct path for manual monkey-patching on the raw Hugging Face model.
    """
    prefix = "base_model.model."
    if peft_name.startswith(prefix):
        return peft_name[len(prefix):]
    return peft_name


###############################
# 5) BaseModelClient
###############################
class BaseModelClient:
    """
    B's side: loads base model, monkey-patches submodules that have LoRA with RemoteLoRAWrappedModule.
    We then can compute token-level loss, which triggers file-based requests to A's side.
    """

    def __init__(self,
                 base_model_name="distilgpt2",
                 file_handler_B=None,
                 combine_mode="replace"):
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_model.eval()

        self.file_handler_B = file_handler_B
        self.combine_mode = combine_mode

    def init_lora_injection_points(self) -> list:
        """
        Sends an init_request to A to get the injection points.
        """
        injection_points = self.file_handler_B.send_init_request()
        print("[B] Received LoRA injection points from A:", injection_points)
        return injection_points

    def inject_remote_lora(self, lora_submodule_names):
        """
        For each submodule in lora_submodule_names, we first fix the name 
        (strip 'base_model.model.'), then replace with RemoteLoRAWrappedModule.
        """
        def _replace_submodule(model: nn.Module, path: str):
            parts = path.split(".")
            *parent_parts, child_name = parts
            m = model
            for p in parent_parts:
                m = getattr(m, p)
            original = getattr(m, child_name)
            wrapped = RemoteLoRAWrappedModule(
                submodule_name=path,
                local_submodule=original,
                file_handler_B=self.file_handler_B,
                combine_mode=self.combine_mode
            )
            setattr(m, child_name, wrapped)
            print(f"[B] Replaced '{path}' with RemoteLoRAWrappedModule.")

        for peft_name in lora_submodule_names:
            # 1) Fix name
            actual_name = fix_peft_submodule_name(peft_name)
            # 2) Attempt replacement
            try:
                _replace_submodule(self.base_model, actual_name)
            except Exception as e:
                print(f"[B] Could not replace '{peft_name}' (mapped to '{actual_name}'): {e}")

    def compute_token_loss(self, tokenizer, text: str):
        """
        Minimal example that does a forward pass with (input_ids, labels=input_ids)
        to compute cross-entropy loss. The replaced submodules do file-based LoRA calls.
        """
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        with torch.no_grad():
            out = self.base_model(input_ids, labels=labels)
            loss = out.loss
        return loss.item()


###############################
# 6) MAIN - Orchestrate the Demo
###############################
def main():
    COMM_FOLDER = "./temp_communication"
    os.makedirs(COMM_FOLDER, exist_ok=True)

    #####################################
    # A) Setup LoRA server
    #####################################
    print("[Main] Initializing LoRA server (User A)...")
    lora_server = LoRAServer(
        base_model_name="distilgpt2",
        lora_model_id="ng0-k1/distilgpt2-finetuned-es"  # local folder or HF Hub
    )

    #####################################
    # B) Start the A-listener thread
    #####################################
    stop_event = threading.Event()
    a_listener_thread = UserAListener(
        lora_server=lora_server,
        comm_folder=COMM_FOLDER,
        stop_event=stop_event
    )
    a_listener_thread.start()

    #####################################
    # C) Setup B's client with file-based requests
    #####################################
    print("[Main] Initializing BaseModelClient (User B)...")
    userB_file_handler = UserBFileHandler(comm_folder=COMM_FOLDER)
    base_client = BaseModelClient(
        base_model_name="distilgpt2",
        file_handler_B=userB_file_handler,
        combine_mode="replace"
    )

    #####################################
    # D) B asks A for LoRA injection points
    #####################################
    injection_points = base_client.init_lora_injection_points()

    # Let's say we only care about c_attn submodules
    submodules_to_patch = [ip for ip in injection_points if "attn.c_attn" in ip]

    print("\n[Main] About to inject remote LoRA for submodules:", submodules_to_patch)
    base_client.inject_remote_lora(submodules_to_patch)

    #####################################
    # E) Compute token-level loss
    #####################################
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    sample_text = "Hello world, this is a LoRA test."
    loss_val = base_client.compute_token_loss(tokenizer, sample_text)
    print(f"[B] Computed token-level loss: {loss_val:.4f}")

    #####################################
    # F) Stop A-listener thread and cleanup
    #####################################
    stop_event.set()  # signal the thread to stop
    a_listener_thread.join()
    print("[Main] A-listener thread stopped.")

    # optional cleanup
    # import shutil
    # shutil.rmtree(COMM_FOLDER, ignore_errors=True)


if __name__ == "__main__":
    main()
