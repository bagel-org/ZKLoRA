import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx
import ezkl
import json
import asyncio
import onnx2torch


# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess MNIST dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Initialize the model
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print(
        f"Test set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.2f}%)"
    )

    # Export the model to ONNX format
    dummy_input = torch.randn(1, 1, 28, 28).to(device)  # MNIST image size is 28x28
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        "mnist_mlp.onnx",  # where to save the model
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )
    print("Model exported to mnist_mlp.onnx")


async def main():
    # Load the ONNX model
    onnx_model = onnx.load("mnist_mlp.onnx")

    # Check the model for consistency
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model is well-formed and valid")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        exit(1)

    #
    # Setup
    #
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"
    ezkl.gen_settings("mnist_mlp.onnx", py_run_args=py_run_args)
    # ezkl.calibrate_settings("mnist_mlp.onnx", "settings.json", target="resources")
    ezkl.compile_circuit("mnist_mlp.onnx", "mnist_mlp.ezkl", "settings.json")
    ezkl.gen_srs("kzg.srs", 17)
    ezkl.setup("mnist_mlp.ezkl", "vk.key", "pk.key", "kzg.srs")

    #
    # Prove
    #
    dummy_input = torch.randn(1, 1, 28, 28)  # MNIST image size is 28x28
    input_data = {
        "input_data": [dummy_input.numpy().reshape(-1).tolist()]  # Flatten the array
    }
    with open("input.json", "w") as f:
        json.dump(input_data, f)
    print("Input data exported to input.json")

    await ezkl.gen_witness(
        data="input.json", model="mnist_mlp.ezkl", output="witness.json"
    )
    
    # Add this code to read and process the witness file
    with open("witness.json", "r") as f:
        witness_data = json.load(f)
    
    # The output is stored in the 'output' field of the witness data
    model_output = witness_data["pretty_elements"]["rescaled_outputs"]
    # Convert the output to a list and get the predicted class
    predictions = model_output[0]  # Get first batch
    predicted_class = max(range(len(predictions)), key=lambda i: predictions[i])
    print("Circuit Model predictions:", predictions)
    print("Circuit Predicted class:", predicted_class)

    # Convert ONNX model to PyTorch
    model = onnx2torch.convert(onnx_model)
    model.eval()  # Set to evaluation mode
    real_model_output = model(dummy_input)
    predictions = real_model_output[0].detach().numpy().tolist()  # Convert tensor to list
    predicted_class = max(range(len(predictions)), key=lambda i: predictions[i])
    print("Real Model predictions:", predictions)
    print("Real Predicted class:", predicted_class)
    
    print("Proving...")
    ezkl.prove(
        witness="witness.json",
        model="mnist_mlp.ezkl",
        pk_path="pk.key",
        proof_path="proof.json",
        srs_path="kzg.srs",
    )
    print("Proof generated")

    print("Verifying...")
    ezkl.verify(
        proof_path="proof.json",
        settings_path="settings.json",
        vk_path="vk.key",
        srs_path="kzg.srs",
    )
    print("Verification complete")

if __name__ == "__main__":
    asyncio.run(main())
