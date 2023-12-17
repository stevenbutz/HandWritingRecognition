import torch
from torch import nn
import torch.optim as optim
import os.path
import Digit_Data



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


    #function used to train the CNN
    @staticmethod
    def train_network(dataloader, output_filename):
        net = NeuralNetwork_Digits()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        global MODEL_PATH
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            #for i, data in enumerate(train_digit_dataloader, 0):
            for i, data in enumerate(dataloader, 0):
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
        torch.save(net.state_dict(), output_filename)
        del net

