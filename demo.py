import random as r
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
import main as m



text_path = "data/testing/"

correct_ones = 0
trys = 0

class Viewer:
    def __init__(self, title="Viewer", image_display_size=(500, 400), resize=False):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(resize, resize)
        self.image_display_size = image_display_size

        self.root.geometry("1500x900")
        self.root.configure(bg="#000000")

        self.image_label = tk.Label(self.root, borderwidth=2, relief="solid", bg = "black")
        self.image_label.pack(padx=20, pady=20)
        self.image_label.place_configure(x=20, y=20)

        self.text_var = tk.StringVar()
        self.text_Label = tk.Label(
            self.root, textvariable=self.text_var,
            font=("Segoe UI", 14), justify="left",
            bg = "black", fg = "white"
        )
        self.text_Label.pack(padx=20, pady=(0, 20))
        self.text_Label.place_configure(x=20, y=430)

        self.num_var = tk.StringVar()
        self.num_Label = tk.Label(
            self.root, textvariable=self.num_var,
            font=("Segoe UI", 11), justify="left",
            bg = "black", fg = "white"
        )
        self.num_Label.pack(padx=20, pady=(0, 20))
        self.num_Label.place_configure(x=436, y=32)

        self.canvas = tk.Canvas(
            self.root,
            width = 920,
            height = 250,
            bg="black",
            bd=0,
            highlightthickness=0
        )
        self.canvas.place_configure(x=450, y=20)

        self.value_var = tk.IntVar(value=1000)
        self.slider = tk.Scale(
            self.root,
            from_=100,
            to=2000,
            resolution=1,
            orient="horizontal",
            variable=self.value_var,
            bg="#000000", fg="white", troughcolor="gray", highlightthickness=0,
            width=20
        )
        self.slider.pack(padx=20, pady=(0, 20))
        self.slider.place_configure(x=200, y=490)
    def update(self, image, text, thrlayer):
        photo = ImageTk.PhotoImage(image.resize((400, 400)))

        self.canvas.delete("all")
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        self.text_var.set(text)
        self.CWidth = self.canvas.winfo_width()
        self.num_var.set("0\n1\n2\n3\n4\n5\n6\n7\n8\n9")
        for LineI in range(thrlayer.size):
            self.canvas.create_line(0, 20 + LineI*20, thrlayer[LineI] * self.CWidth, LineI * 20 + 20, fill="green", width=6)




    def run(self):
        self.root.mainloop()



def example_auto_updating():
    def predict_fn(_img, num):
        global trys
        global correct_ones
        pixels = np.array(_img, dtype=np.float32).flatten()
        input_neuron = m.normalize_pixel(pixels)
        raw_secLayer = np.dot(input_neuron, firWeight) - firBais
        secLayer = m.sigma(raw_secLayer)
        raw_thrLayer = np.dot(secLayer, secWeight) - secBais
        thrLayer = m.softMax(raw_thrLayer)
        d = np.argmax(thrLayer)
        if num == d:
            correct_ones += 1
        trys += 1
        return d, correct_ones / trys, thrLayer

    viewer = Viewer(title="Predictor", resize=True, image_display_size=(500, 400))

    def tick():
        global correct_ones
        global trys
        global text_path
        test_num = -1
        while True:
            test_num = r.randint(0, 9)
            text_path += str(test_num)
            file_path = Path(text_path + "/" + str(r.randint(0, 60000)) + ".png")
            if file_path.is_file():
                img = Image.open(file_path).convert('L')
                break
            text_path = text_path.replace(str(test_num), "")
        label, confidence, ThrLayer = predict_fn(img, test_num)
        viewer.update(img, f"Predicted: {label}\nActual Num: {test_num} \n Accuracy : {confidence * 100:.1f}%\nUpdate time:", ThrLayer)
        viewer.root.after(viewer.value_var.get(), tick)

    viewer.root.after(10, tick)
    viewer.run()


if __name__ == "__main__":
    firBais = np.load(Path("savedData/B1.npy"))
    secBais = np.load(Path("savedData/B2.npy"))
    firWeight = np.load(Path("savedData/W1.npy"))
    secWeight = np.load(Path("savedData/W2.npy"))
    example_auto_updating()