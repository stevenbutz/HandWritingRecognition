import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageDraw
import CNN_Network
from torchvision import transforms
import torch
import os.path
import feedforeward
import torch
import Digit_Data
import Metric_Calculations
import Drawing_Classification_Demo
import Testing_Sample_Classification_Demo

ff_EMNIST_filename = os.path.join(Digit_Data.script_dir, 'ff_EMNIST.pth')
cnn_EMNIST_filename = os.path.join(Digit_Data.script_dir, 'cnn_EMNIST.pth')
ff_MNIST_filename = os.path.join(Digit_Data.script_dir, 'ff_MNIST.pth')
cnn_MNIST_filename = os.path.join(Digit_Data.script_dir, 'cnn_MNIST.pth')

def check_for_trained_network(Model_Path, network, dataloader):
    try:
        ii= torch.load(Model_Path)
        network.load_state_dict(ii)
    except IOError:
        print("Could not find saved network file. Would you like to train the network?")
        answer=input("Enter y for Yes or n for No")
        if answer=="y":
            network.train_network(dataloader, Model_Path)
        elif answer=="n":
            raise Exception("Can't run without trained network. Exiting program.")
        else:
            raise Exception("Can't read answer. Exiting program.")


if __name__=="__main__":
    ff_model=feedforeward.SimpleNN()
    cnn_model=CNN_Network.NeuralNetwork_Digits()

    check_for_trained_network(ff_EMNIST_filename, ff_model, Digit_Data.train_digit_dataloader_EMNIST)
    check_for_trained_network(cnn_EMNIST_filename, cnn_model, Digit_Data.train_digit_dataloader_EMNIST)

    ff_model.eval()
    cnn_model.eval()

    print("Retrain networks? y/n")
    t=input()
    if t=='y':
        print("Training Data")
        cnn_model.train_network(Digit_Data.train_digit_dataloader_EMNIST, cnn_EMNIST_filename)
        ff_model.train_network(Digit_Data.train_digit_dataloader_EMNIST, ff_EMNIST_filename)
    else:
        print("Skipping Training")

    print("Check Accuracy? y/n")
    t=input()
    if t=='y':
        print("Checking Accuracy")
        Metric_Calculations.print_accuracy(Metric_Calculations.check_accuracy(cnn_model, Digit_Data.test_digit_dataloader_EMNIST))
        Metric_Calculations.print_accuracy(Metric_Calculations.check_accuracy(ff_model, Digit_Data.test_digit_dataloader_EMNIST))
        #check_accuracy()
    else:
        print("Skipping Accuracy Check")
    print("Run Drawing Demo? y/n")
    t=input()
    if t=='y':
        print("Running Drawing Demo")
        Drawing_Classification_Demo.Image_Classifier_Demo(cnn_model,ff_model)
    else:
        print("Skipping Known Digit Classification Demo")
    print("Run Training Data Demo? y/n")
    t=input()
    if t=='y':
        print("Running Demo")
        Testing_Sample_Classification_Demo.known_digit_classification_demo(Digit_Data.test_data_digits_EMNIST,ff_model,cnn_model)
    else:
        print("Skipping Training Data Demo")