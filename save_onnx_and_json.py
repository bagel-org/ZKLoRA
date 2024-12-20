import os
import json
import numpy as np
import torch
import torch.nn as nn
import onnx
from torch.onnx import TrainingMode

#############################################
# Configuration
#############################################
# Assume we have saved A.npy and B.npy already
A_data = np.load("lora_params/A.npy")
B_data = np.load("lora_params/B.npy")

# Assume x_data was captured previously
x_data = np.load("x_data.npy")  # For example, if you saved it previously

batch_size, seq_len, hidden_dim = x_data.shape

#############################################
# Define a model with A and B as parameters
#############################################
class LoraApplyModel(nn.Module):
    def __init__(self, A_data, B_data):
        super().__init__()
        # Register A and B as parameters so they are known to the model
        # but since we will use export_params=False, they won't be embedded.
        A = torch.from_numpy(A_data).float()
        B = torch.from_numpy(B_data).float()
        # Use nn.Parameter so they are parameters of the model
        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, x):
        in_dim = x.shape[-1]
        B = self.B
        A = self.A

        # Fix shapes if needed
        if B.shape[0] != in_dim:
            B = B.transpose(0, 1)
        if A.shape[0] != B.shape[1]:
            A = A.transpose(0, 1)

        out = torch.matmul(torch.matmul(x, B), A)
        # Depend on x, A, B
        out = out + x.mean() + A.sum() + B.sum()
        return out

model_for_onnx = LoraApplyModel(A_data, B_data).eval()
x_tensor = torch.from_numpy(x_data)

onnx_path = "lora_onnx_params/lora_layer0_params.onnx"

torch.onnx.export(
    model_for_onnx,
    x_tensor,
    onnx_path,
    export_params=False,        # not embedding params as constants
    do_constant_folding=False,
    opset_version=11,
    input_names=["input_x"],
    output_names=["output"],
    dynamic_axes={
        "input_x": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    },
    training=TrainingMode.TRAINING
)

# Check inputs
onnx_model = onnx.load(onnx_path)
print("ONNX Inputs:", onnx_model.graph.input)

# If you see only `input_x` and not A and B as separate inputs, it means A and B are still treated as constants/initializers.
# In that case, you'd revert to passing A and B as inputs to forward().
