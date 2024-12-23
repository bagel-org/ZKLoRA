import torch
import torch.nn as nn
import onnx

class SimpleMultiInputModel(nn.Module):
    def forward(self, x, A, B):
        # Perform a computation that depends on all inputs:
        # x: (B, S, in_dim)
        # A, B: assume (in_dim, in_dim) for simplicity
        out = torch.matmul(x, B)     # (B, S, in_dim)
        out = torch.matmul(out, A)   # (B, S, in_dim)
        out = out + x.mean()         # depends on x values
        return out

model = SimpleMultiInputModel().eval()

# Create dummy inputs
x = torch.randn(1, 5, 768)
A = torch.randn(768, 768)
B = torch.randn(768, 768)

onnx_path = "test_multi_input.onnx"

torch.onnx.export(
    model,
    (x, A, B),
    onnx_path,
    export_params=False,        # do not embed parameters as constants
    do_constant_folding=False,  # do not fold constants
    opset_version=11,
    input_names=["input_x", "input_A", "input_B"],
    output_names=["output"],
    dynamic_axes={
        "input_x": {0: "batch_size", 1: "seq_len"},
        "output": {0: "batch_size", 1: "seq_len"}
    }
)

onnx_model = onnx.load(onnx_path)
print("ONNX Inputs:", onnx_model.graph.input)
# Expected: Should see input_x, input_A, input_B as inputs.
