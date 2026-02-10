import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("model/deforestation_model.h5")
class_names = ["deforested", "forest"]
img_size = 224

def upload_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if img_path:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((250,250))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")

def predict_image():
    if not img_path:
        return
    img = Image.open(img_path).convert("RGB")
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[np.argmax(prediction)]
    result_label.config(text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")

root = tk.Tk()
root.title("Deforestation Detection")
root.geometry("400x500")

title_label = tk.Label(root, text="Deforestation Detection System", font=("Arial", 14))
title_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

upload_btn = tk.Button(root, text="Upload Image", command=upload_image, width=20)
upload_btn.pack(pady=5)

predict_btn = tk.Button(root, text="Predict", command=predict_image, width=20)
predict_btn.pack(pady=5)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack(pady=15)

img_path = None
root.mainloop()
