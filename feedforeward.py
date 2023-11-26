import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define a feedforward network with two hidden layers
        self.fc1 = nn.Linear(784, 128) # First hidden layer
        self.fc2 = nn.Linear(128, 64)  # New second hidden layer
        self.fc3 = nn.Linear(64, 10)   # Output layer

    def forward(self, x):
        # Flatten the image tensor
        x = x.view(-1, 28 * 28)
        # Apply the first fully connected layer with ReLU activation function
        x = torch.relu(self.fc1(x))
        # Apply the second fully connected layer with ReLU activation function
        x = torch.relu(self.fc2(x))
        # Output layer with logits, no activation function
        x = self.fc3(x)
        return x


# Load MNIST dataset
transform = transforms.ToTensor()
full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Calculate sizes for split (90% train, 10% test)
train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

# Split the dataset 
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Data loaders for training and testing datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)



# Initialize the neural network, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(3):  # Enter number for how many epoch 
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

#Test the model 

model.eval()

correct = 0
total = 0

# Disabling gradient calculation
with torch.no_grad():
    for data, target in test_loader:
        # Forward pass
        output = model(data)

        # Get the index of the max log-probability
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

# Calculate and print the accuracy
accuracy = 100 * correct / total
print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

