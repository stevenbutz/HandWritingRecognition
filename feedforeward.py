import torch
import torch.nn as nn
import torch.optim as optim
import os
import Digit_Data



# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define a feedforward network with two hidden layers
        self.fc1 = nn.Linear(784, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)   # Second hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer

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


    @staticmethod
    def train_network(dataloader, output_filename):
        model = SimpleNN()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        # Training loop
        for epoch in range(3):  # Enter number for how many epochs
            for batch_idx, (data, target) in enumerate(dataloader):
                # Forward pass
                output = model(data)
                loss = criterion(output, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}")
        MODEL_PATH = os.path.join(Digit_Data.script_dir, output_filename)
        torch.save(model.state_dict(), MODEL_PATH)
        del model

