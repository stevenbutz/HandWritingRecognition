import tkinter as tk
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageDraw
import pytorch_test
from torchvision import transforms
import torch
import os.path

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
    output = pytorch_test.net(image_tensor.unsqueeze(1))
    test=torch.softmax(output, 1, float)
    #test=torch.log_softmax(output)
    print(test)
    print(output)
    _, predicted = torch.max(output, 1)
    Predicted_StringVar.set(f"Predicted: {predicted.item()}")
    Percentages_Text.configure(state=NORMAL)
    Percentages_Text.delete(1.0, END)
    output_String="\n".join(f"Digit {x}: {round(test[0][x].item(), 3)}" for x in range(10)) 
    Percentages_Text.insert(1.0, output_String)
    Percentages_Text.configure(state=DISABLED)

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

Predicted_StringVar = tk.StringVar(value="Predicted: Default")
Predicted_Label = ttk.Label(CNN_Frame,textvariable=Predicted_StringVar, font=("Arial", 25))
#Percentages_Text_StringVar = tk.StringVar(value="Default")
Percentages_Text = tk.Text(CNN_Frame, state="disabled")
#Percentages_Text = tk.Text(CNN_Frame)
Predicted_Label.grid(column=0, row=0, sticky=(N,W,E,S))
Percentages_Text.grid(column=1, row=0, sticky=(N,W,E,S))


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