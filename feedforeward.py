import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
#PATH = '/home/dan/digits_net.pth'
MODEL_PATH = os.path.join(script_dir, 'digits_ff_model.pth')
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

def train():
    # Training loop
    for epoch in range(3):  # Enter number for how many epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
    torch.save(model.state_dict(), MODEL_PATH)
# Test the model 
#train()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Initialize arrays to store predictions
correct_pred = {classname: 0 for classname in range(10)}
total_pred = {classname: 0 for classname in range(10)}
true_positives = {classname: 0 for classname in range(10)}
false_positives = {classname: 0 for classname in range(10)}
false_negatives = {classname: 0 for classname in range(10)}

# Test the model
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predictions = torch.max(output, 1)
        for label, prediction in zip(target, predictions):
            total_pred[label.item()] += 1  # Update total predictions for each class
            if label == prediction:
                correct_pred[label.item()] += 1
                true_positives[label.item()] += 1
            else:
                false_positives[prediction.item()] += 1
                false_negatives[label.item()] += 1

# Overall accuracy
total_correct = sum(correct_pred.values())
total = sum(total_pred.values())
overall_accuracy = 100 * total_correct / total

# Calculate and print overall accuracy, precision, recall, and F1 score for each class
for classname in range(10):
    # Calculating metrics for each class
    class_precision = true_positives[classname] / max((true_positives[classname] + false_positives[classname]), 1)
    class_recall = true_positives[classname] / max((true_positives[classname] + false_negatives[classname]), 1)
    class_f1 = 2 * (class_precision * class_recall) / max((class_precision + class_recall), 1)

    # Calculating and printing class-wise accuracy
    class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
    #print(f'Accuracy for class: {classname}     is {class_accuracy:.1f} %')
    # Printing precision, recall, and F1 score for each class, multiplied by 100 for percentage
    #print(f'Precision for class: {classname}     is {class_precision * 100:.1f} %')
    #print(f'Recall for class: {classname}     is {class_recall * 100:.1f} %')
    #print(f'F1 for class: {classname}     is {class_f1 * 100:.1f} %')
    #print()  # Adds a blank line for readability
    print(f'{class_accuracy:.1f},{class_precision:.1f},{class_recall:.1f},{class_f1:.1f}')
# Print overall accuracy
print(f'Overall Accuracy of the network on the test images: {overall_accuracy:.1f}%')