import tkinter as tk
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

class DigitRecognizer(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        # Load the pre-trained model
        self.model = load_model('num_predict.model')

        # Create the drawing canvas
        self.canvas = tk.Canvas(self, width=280, height=280, bg = "white", cursor="cross")
        self.title("HandWritten Digit Recognizer GUI")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 20))
        self.button_recognize = tk.Button(self, text = "Recognize", command = self.recognize_digit)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_canvas)

        # Arrange the elements
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=2, column=0, pady=2, padx=2, columnspan=2)
        self.button_recognize.grid(row=1, column=0, pady=2)
        self.button_clear.grid(row=1, column=1, pady=2)

        self.canvas.bind("<B1-Motion>", self.draw_lines)

        self.drawing = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.drawing)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=6
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
        self.draw.ellipse([self.x-r, self.y-r, self.x + r, self.y + r], fill='black')

    def recognize_digit(self):
        img = self.drawing.resize((28, 28))
        img = img.convert('L')
        img=np.invert(np.array(img))
        img = np.array(img)
        img = img.reshape(1, 28, 28, 1)
        img = img/255.0

        res = self.model.predict([img])[0]
        digit=np.argmax(res)
        acc = max(res)*100
        self.label.configure(text= 'digit: '+ str(digit)+', Probability '+ str(int(acc))+'%')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing = Image.new("RGB", (280, 280), "white")
        self.label.configure(text="Draw..")
        self.draw = ImageDraw.Draw(self.drawing)

app = DigitRecognizer()
app.mainloop()