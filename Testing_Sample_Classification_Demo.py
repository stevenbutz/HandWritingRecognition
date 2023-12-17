import torch
import matplotlib.pyplot as plt
import sys

def known_digit_classification_demo(testdata,ff_model, cnn_model):
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
            sample_idx = torch.randint(len(testdata), size=(1,)).item()
            img, label = testdata[sample_idx]
            ff_output = ff_model(img.unsqueeze(1))
            _, predicted_ff = torch.max(ff_output, 1)
            cnn_output = cnn_model(img.unsqueeze(1))
            _, predicted_cnn = torch.max(cnn_output, 1)
            predicted_cnn_value=predicted_cnn.item()
            predicted_ff_value=predicted_ff.item()
            figure.add_subplot(rows, cols, i)
            plt.title(f"Known Label:{label}\nCNN:{predicted_cnn_value}\nFF:{predicted_ff_value}")
            plt.axis("off")
            plt.imshow(img.squeeze(), cmap="gray")
    build()
    figure.tight_layout(h_pad=2)
    figure.canvas.mpl_connect('key_press_event', on_press)
    plt.show()

