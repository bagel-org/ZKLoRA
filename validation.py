import ezkl

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),  # First layer: 784 (28x28 pixels) -> 128
            nn.ReLU(),
            nn.Linear(128, 10)        # Second layer: 128 -> 10 (number of digits)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

# Load and preprocess MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                         batch_size=64,
                                         shuffle=True)

# Initialize model, loss function and optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print("Training finished!")

# Load validation dataset
validation_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transform,
                                              download=True)

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                              batch_size=64,
                                              shuffle=False)

# Validation
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in validation_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

model.train()  # Set the model back to training mode
