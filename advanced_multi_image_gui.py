import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

img_size = 224
model = tf.keras.models.load_model("model/deforestation_model.h5")
class_names = ["deforested", "forest"]

image_paths = []
image_widgets = []

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def upload_images():
    global image_paths
    image_paths = filedialog.askopenfilenames(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    clear_results()
    display_images()

def display_images():
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((120, 120))
        photo = ImageTk.PhotoImage(img)

        frame = tk.Frame(result_frame, padx=10, pady=10, relief="ridge", borderwidth=2)
        frame.pack(side="left", padx=5)

        label_img = tk.Label(frame, image=photo)
        label_img.image = photo
        label_img.pack()

        label_text = tk.Label(frame, text="Not Predicted", font=("Arial", 10))
        label_text.pack()

        image_widgets.append((frame, label_text, path))

def predict_images():
    for frame, label_text, path in image_widgets:
        img_array = preprocess_image(path)
        prediction = model.predict(img_array, verbose=0)[0]
        class_index = np.argmax(prediction)
        confidence = prediction[class_index] * 100
        label = class_names[class_index]

        color = "green" if label == "forest" else "red"
        label_text.config(
            text=f"{label.upper()}\n{confidence:.2f}%",
            fg=color
        )

def clear_results():
    global image_widgets
    for widget in result_frame.winfo_children():
        widget.destroy()
    image_widgets = []

root = tk.Tk()
root.title("Advanced Deforestation Detection System")
root.geometry("900x500")

title = tk.Label(root, text="Deforestation Detection System", font=("Arial", 18))
title.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

upload_btn = tk.Button(btn_frame, text="Upload Images", command=upload_images, width=15)
upload_btn.pack(side="left", padx=10)

predict_btn = tk.Button(btn_frame, text="Predict", command=predict_images, width=15)
predict_btn.pack(side="left", padx=10)

canvas = tk.Canvas(root)
canvas.pack(fill="both", expand=True)

scrollbar = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
scrollbar.pack(side="bottom", fill="x")

canvas.configure(xscrollcommand=scrollbar.set)
result_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=result_frame, anchor="nw")

result_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

root.mainloop()
