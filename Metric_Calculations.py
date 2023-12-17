import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageDraw
import CNN_Network
from torchvision import transforms
import torch
import os.path
import feedforeward
import sys
import matplotlib.pyplot as plt
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
import Digit_Data

# since we're not training, we don't need to calculate the gradients for our outputs
def check_accuracy(model, dataloader):
    digit_res={}
    TP = {str(i): 0 for i in range(10)}
    FP = {str(i): 0 for i in range(10)}
    TN= {str(i): 0 for i in range(10)}
    FN = {str(i): 0 for i in range(10)}
    total_pred = {str(i): 0 for i in range(10)}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
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
                    FN[str(label.item())] +=1
                    for k, v in TN.items():
                        if k != label.item() or k != prediction.item():
                            TN[k] += 1

                total_pred[str(label.item())] += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    for classname in range(10):
        classname=str(classname)
        precision = 100 * float(TP[classname]) / (TP[classname] + FP[classname])
        accuracy = 100 * float(TP[classname]) / total_pred[classname]
        recall = 100 * float(TP[classname]) / (TP[classname] + FN[classname])
        f1 = 2 * (accuracy * recall) / (accuracy + recall)
        digit_res[str(classname)]={"f1":f1,"precision":precision,"recall":recall,"accuracy":accuracy}
    digit_res["total_accuracy"]=100 * correct // total
    return digit_res
def print_accuracy(data):
    for i in range(10):
        classname=str(i)
        print(f'Accuracy for class: {classname:5s} is {data[classname]["accuracy"]:.1f} %')
        print(f'Precision for class: {classname:5s} is {data[classname]["precision"]:.1f} %')
        print(f'Recall for class: {classname:5s} is {data[classname]["recall"]:.1f} %')
        print(f'F1 for class: {classname:5s} is {data[classname]["f1"]:.1f} %')
        print()
        print(f'Accuracy of the network on the 10000 test images: {data["total_accuracy"]} %')
                #print(f'{accuracy:.1f},{precision:.1f},{recall:.1f},{f1:.1f}')

# Initialize arrays to store predictions
#correct_pred = {classname: 0 for classname in range(10)}
#total_pred = {classname: 0 for classname in range(10)}
#true_positives = {classname: 0 for classname in range(10)}
#false_positives = {classname: 0 for classname in range(10)}
#false_negatives = {classname: 0 for classname in range(10)}
#
## Test the model
#with torch.no_grad():
#    for data, target in test_loader:
#        output = model(data)
#        _, predictions = torch.max(output, 1)
#        for label, prediction in zip(target, predictions):
#            total_pred[label.item()] += 1  # Update total predictions for each class
#            if label == prediction:
#                correct_pred[label.item()] += 1
#                true_positives[label.item()] += 1
#            else:
#                false_positives[prediction.item()] += 1
#                false_negatives[label.item()] += 1
#
## Overall accuracy
#total_correct = sum(correct_pred.values())
#total = sum(total_pred.values())
#overall_accuracy = 100 * total_correct / total
#
## Calculate and print overall accuracy, precision, recall, and F1 score for each class
#for classname in range(10):
#    # Calculating metrics for each class
#    class_precision = true_positives[classname] / max((true_positives[classname] + false_positives[classname]), 1)
#    class_recall = true_positives[classname] / max((true_positives[classname] + false_negatives[classname]), 1)
#    class_f1 = 2 * (class_precision * class_recall) / max((class_precision + class_recall), 1)
#
#    # Calculating and printing class-wise accuracy
#    class_accuracy = 100 * correct_pred[classname] / total_pred[classname]
#    #print(f'Accuracy for class: {classname}     is {class_accuracy:.1f} %')
#    # Printing precision, recall, and F1 score for each class, multiplied by 100 for percentage
#    #print(f'Precision for class: {classname}     is {class_precision * 100:.1f} %')
#    #print(f'Recall for class: {classname}     is {class_recall * 100:.1f} %')
#    #print(f'F1 for class: {classname}     is {class_f1 * 100:.1f} %')
#    #print()  # Adds a blank line for readability
#    print(f'{class_accuracy:.1f},{class_precision:.1f},{class_recall:.1f},{class_f1:.1f}')
## Print overall accuracy
#print(f'Overall Accuracy of the network on the test images: {overall_accuracy:.1f}%')

