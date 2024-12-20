import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import onnx

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
    
def train():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

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
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    print(f'Test set: Average loss: {test_loss:.4f}, '
            f'Accuracy: {correct}/{len(test_loader.dataset)} '
            f'({100. * correct / len(test_loader.dataset):.2f}%)')
    
    # Export the model to ONNX format
    dummy_input = torch.randn(1, 1, 28, 28).to(device)  # MNIST image size is 28x28
    torch.onnx.export(
        model,                     # model being run
        dummy_input,              # model input (or a tuple for multiple inputs)
        "mnist_mlp.onnx",         # where to save the model
        export_params=True,       # store the trained parameter weights inside the model file
        opset_version=10,         # the ONNX version to export the model to
        do_constant_folding=True, # whether to execute constant folding for optimization
        input_names=['input'],    # the model's input names
        output_names=['output'],  # the model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},  # variable length axes
            'output': {0: 'batch_size'}
        }
    )
    print("Model exported to mnist_mlp.onnx")

if __name__ == "__main__":
    
    # Load the ONNX model
    onnx_model = onnx.load("mnist_mlp.onnx")
    
    # Check the model for consistency
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model is well-formed and valid")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
        exit(1)
    

    


