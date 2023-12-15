import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import numpy as np
from os.path import realpath
import os.path
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
#PATH = '/home/dan/digits_net.pth'
MODEL_PATH = os.path.join(script_dir, 'digits_net.pth')


training_data_digits = datasets.EMNIST(
    root="data",
    train=True,
    split='digits',
    download=True,
    transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()])
)

test_data_digits = datasets.EMNIST(
    root="data",
    train=False,
    split='digits',
    download=True,
    transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()])
)


train_digit_dataloader = DataLoader(training_data_digits, batch_size=24, shuffle=True)
test_digit_dataloader = DataLoader(test_data_digits, batch_size=24, shuffle=True)

class NeuralNetwork_Digits(nn.Module):
    def __init__(self):
        super().__init__()
        #input images are 28*28 pixels and have only 1 color channel
        self.digits_model = nn.Sequential(
            #takes a one channel image (gray) and runs 6 different 3x3 learnable kernals to create 6 28*28 feature maps
            nn.Conv2d(1,6,3,padding=1),
            #the ReLU activation function helps with non-linearity
            nn.ReLU(),
            #the max pooling reduces the height and width of our feature maps to save computation time and memory
            #our pooling layer has a kernal of 2*2 which should reduce  our feature maps to 14*14*6
            nn.MaxPool2d(2,2),
            
            #takes our 6 14*14 feature maps from the max pooling layer and creates a new 12 14*14 feature maps
            nn.Conv2d(6,12,3, padding=1),
            #the ReLU activation function helps with non-linearity
            nn.ReLU(),
            #the max pooling reduces the height and width of our feature maps to save computation time and memory
            #this pooling layer has a kernal of 2*2 which should reduce  our feature maps to 7*7*12
            nn.MaxPool2d(2,2),

            #reduces our 3d feature map to a single dimesion to feed into the linear network
            nn.Flatten(),

            #standard feed forward neural network used to classify the ouput of the convolution and pooling layers
            nn.Linear(7*7*12, 120),
            #the ReLU activation function helps with non-linearity
            nn.ReLU(),
            #final layer in the network with 10 ouput nodes that represent the 10 possible classification digits
            nn.Linear(120, 10),
        )

    def forward(self, x):
        return self.digits_model(x)

net = NeuralNetwork_Digits()

#function used to train the CNN
def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    global MODEL_PATH
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        #for i, data in enumerate(train_digit_dataloader, 0):
        for i, data in enumerate(train_digit_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training') 
    torch.save(net.state_dict(), MODEL_PATH)

try:
    net.load_state_dict(torch.load(MODEL_PATH))
except IOError:
    print("Could not find saved network file. Would you like to train the network?")
    answer=input("Enter y for Yes or n for No")
    if answer=="y":
        train()
    elif answer=="n":
        raise Exception("Can't run without trained network. Exiting program.")
    else:
        raise Exception("Can't read answer. Exiting program.")

dataiter = iter(test_digit_dataloader)
images, labels = next(dataiter)

# since we're not training, we don't need to calculate the gradients for our outputs
def check_accuracy():
    print("Checking accuracy of CNN.")
    TP = {str(i): 0 for i in range(10)}
    FP = {str(i): 0 for i in range(10)}
    TN= {str(i): 0 for i in range(10)}
    FN = {str(i): 0 for i in range(10)}
    total_pred = {str(i): 0 for i in range(10)}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_digit_dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    TP[str(label.item())] += 1
                    for k, v in TN.items():
                        if label != prediction:
                            TN[k] += 1
                else:
                    FP[str(prediction.item())] +=1
                    for k, v in FN.items():
                        if label != prediction:
                            FN[k] += 1

                total_pred[str(label.item())] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    for classname, correct_count in TP.items():
        precision = 100 * float(TP[classname]) / (TP[classname] + FP[classname])
        accuracy = 100 * float(correct_count) / total_pred[classname]
        recall = 100 * float(TP[classname]) / (TP[classname] + FN[classname])
        f1 = 2 * (accuracy * recall) / (accuracy + recall)
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        print(f'Precision for class: {classname:5s} is {precision:.1f} %')
        print(f'Recall for class: {classname:5s} is {recall:.1f} %')
        print(f'F1 for class: {classname:5s} is {f1:.1f} %')
        print()
        #print(f'{accuracy:.1f},{precision:.1f},{recall:.1f},{f1:.1f}')

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def known_digit_classification_demo():
    figure = plt.figure(figsize=(8, 8))
    def on_press(event):
        sys.stdout.flush()
        if event.key == 'enter':
            for i in figure.get_axes():
                figure.delaxes(i)
            build()
            plt.draw()
        if event.key == 'escape':
            plt.close()
    def build():
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(test_data_digits), size=(1,)).item()
            img, label = test_data_digits[sample_idx]
            output = net(img.unsqueeze(1))
            _, predicted = torch.max(output, 1)
            predicted_value=predicted.item()
            figure.add_subplot(rows, cols, i)
            if label==predicted_value:
                plt.rcParams["axes.titlecolor"]="green"
            else:
                plt.rcParams["axes.titlecolor"]="red"
            plt.title(f"Known Label:{label}\nPredicted Label:{predicted_value}")
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
    build()
    figure.tight_layout(h_pad=2)
    figure.canvas.mpl_connect('key_press_event', on_press)
    plt.show()

if __name__=="__main__":
    print("Train Data? y/n")
    t=input()
    if t=='y':
        print("Training Data")
        train()
    else:
        print("Skipping Training")

    print("Check Accuracy? y/n")
    t=input()
    if t=='y':
        print("Checking Accuracy")
        check_accuracy()
    else:
        print("Skipping Accuracy Check")
    print("Run Known Digit Classification Demo? y/n")
    t=input()
    if t=='y':
        print("Running Known Digit Classification Demo")
        known_digit_classification_demo()
    else:
        print("Skipping Known Digit Classification Demo")

