import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageDraw
from CNN import pytorch_test
from torchvision import transforms
import torch
import os.path
import feedforeward


TEST_IMAGE_PATH = os.path.join(pytorch_test.script_dir, 'test_Image.png')
root = Tk()
CNN_Frame = ttk.Frame(root)
Feed_Forward_Frame = ttk.Frame(root)
Canvas_Frame = ttk.Frame(root)
canvas_scale_size=20

image1 = Image.new("L", (28, 28))
draw = ImageDraw.Draw(image1)

def classify_digit():
    image1.save(TEST_IMAGE_PATH)
    image_tensor = transforms.functional.pil_to_tensor(image1).float()
    cnn_output = pytorch_test.net(image_tensor.unsqueeze(1))
    ff_output = feedforeward.model(image_tensor.unsqueeze(1))
    test=torch.softmax(cnn_output, 1, float)
    test=torch.softmax(cnn_output, 1, float)
    #test=torch.log_softmax(output)
    print(test)
    print(cnn_output)
    _, cnn_predicted = torch.max(cnn_output, 1)
    _, ff_predicted = torch.max(ff_output, 1)
    Predicted_StringVar.set(f"Predicted: {cnn_predicted.item()}")
    ff_Predicted_StringVar.set(f"Predicted: {ff_predicted.item()}")
    output_String="\n".join(f"Digit {x}: {round(test[0][x].item(), 3)}" for x in range(10)) 

def clear_canvas():
    global image1, draw
    del image1
    del draw
    canvas.delete("all")
    image1 = Image.new("L", (28, 28))
    draw = ImageDraw.Draw(image1)


canvas = Canvas(Canvas_Frame, width=canvas_scale_size*28, height=canvas_scale_size*28, background="black")
Classify_Button = ttk.Button(Canvas_Frame,text="Classify", command=classify_digit)
Clear_Canvas_Button = ttk.Button(Canvas_Frame,text="Clear", command=clear_canvas)

CNN_Frame.grid(column=0, row=0, sticky=(N,W,E,S))
Feed_Forward_Frame.grid(column=1, row=0, sticky=(N,W,E,S))
Canvas_Frame.grid(column=0, row=1, columnspan=2, sticky=(N,W,E,S))

canvas.grid(column=0, row=0, columnspan=2, sticky=(N,W,E,S))
Classify_Button.grid(column=0, row=1, sticky=(N,W,E,S))
Clear_Canvas_Button.grid(column=1, row=1, sticky=(N,W,E,S))

root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

ff_Predicted_StringVar = tk.StringVar(value="Predicted: Default")
ff_Predicted_Label = ttk.Label(Feed_Forward_Frame,textvariable=ff_Predicted_StringVar, font=("Arial", 25))
ff_Predicted_Title_Label = ttk.Label(Feed_Forward_Frame,text="Feed Forward", font=("Arial", 25))
#Percentages_Text_StringVar = tk.StringVar(value="Default")
#Percentages_Text = tk.Text(CNN_Frame)
ff_Predicted_Title_Label.grid(column=0, row=0, sticky=(N,W,E,S))
ff_Predicted_Label.grid(column=0, row=1, sticky=(N,W,E,S))

Predicted_StringVar = tk.StringVar(value="Predicted: Default")
Predicted_Label = ttk.Label(CNN_Frame,textvariable=Predicted_StringVar, font=("Arial", 25))
Predicted_Title_Label = ttk.Label(CNN_Frame,text="Convolution Network", font=("Arial", 25))
#Percentages_Text_StringVar = tk.StringVar(value="Default")
#Percentages_Text = tk.Text(CNN_Frame)
Predicted_Title_Label.grid(column=0, row=0, sticky=(N,W,E,S))
Predicted_Label.grid(column=0, row=1, sticky=(N,W,E,S))


lastx, lasty = 0, 0


def xy(event):
    global lastx, lasty
    lastx, lasty = canvas.canvasx(event.x), canvas.canvasy(event.y)


def FillPixle(event):
    global lastx, lasty
    x, y = canvas.canvasx(event.x), canvas.canvasy(event.y)
    x_canvas_location=(x//canvas_scale_size)*canvas_scale_size
    y_canvas_location=(y//canvas_scale_size)*canvas_scale_size
    canvas.create_rectangle((x_canvas_location, y_canvas_location, x_canvas_location+canvas_scale_size,y_canvas_location+canvas_scale_size), fill="white",)
    x_pixel=x//canvas_scale_size
    y_pixel=y//canvas_scale_size
    draw.rectangle([x_pixel,y_pixel,x_pixel+1,y_pixel+1], 128)
    lastx, lasty = x, y

def doneStroke(event):
    canvas.itemconfigure('currentline', width=10)        
        
canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", FillPixle)


root.mainloop()