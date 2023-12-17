import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import os.path
from torch.utils.data import DataLoader, random_split

script_dir = os.path.dirname(os.path.abspath(__file__))

training_data_digits_EMNIST = datasets.EMNIST(
    root="data",
    train=True,
    split='digits',
    download=True,
    transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()])
)

test_data_digits_EMNIST = datasets.EMNIST(
    root="data",
    train=False,
    split='digits',
    download=True,
    transform=torchvision.transforms.Compose([
                    lambda img: torchvision.transforms.functional.rotate(img, -90),
                    lambda img: torchvision.transforms.functional.hflip(img),
                    torchvision.transforms.ToTensor()])
)


train_digit_dataloader_EMNIST = DataLoader(training_data_digits_EMNIST, batch_size=24, shuffle=True)
test_digit_dataloader_EMNIST = DataLoader(test_data_digits_EMNIST, batch_size=24, shuffle=True)

# Load MNIST dataset
transform = transforms.ToTensor()
full_dataset_MNIST = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Calculate sizes for split (90% train, 10% test)
train_size = int(0.9 * len(full_dataset_MNIST))
test_size = len(full_dataset_MNIST) - train_size

# Split the dataset 
train_dataset_MNIST, test_dataset_MNIST = random_split(full_dataset_MNIST, [train_size, test_size])

# Data loaders for training and testing datasets
train_loader_MNIST = DataLoader(train_dataset_MNIST, batch_size=64, shuffle=True)
test_loader_MNIST = DataLoader(test_dataset_MNIST, batch_size=1000, shuffle=False)

