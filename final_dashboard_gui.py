import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.preprocessing import image
import os

MODEL_PATH = "model/deforestation_model.h5"
IMG_SIZE = (224, 224)
class_names = ["deforested", "forest"]

model = tf.keras.models.load_model(MODEL_PATH)

root = tk.Tk()
root.title("Deforestation Detection System")
root.geometry("1400x800")
root.configure(bg="black")

uploaded_images = []
y_true = []
y_pred = []
confidences = []

# ---------- TITLE ----------
tk.Label(root, text="Deforestation Detection System",
         font=("Arial", 24, "bold"),
         fg="white", bg="black").pack(pady=10)

# ---------- CONTROLS ----------
control_frame = tk.Frame(root, bg="black")
control_frame.pack(pady=10)

actual_label = tk.StringVar(value="forest")

ttk.Label(control_frame, text="Actual Class:",
          background="black", foreground="white").pack(side=tk.LEFT, padx=5)

ttk.Combobox(control_frame, textvariable=actual_label,
             values=class_names, width=12, state="readonly").pack(side=tk.LEFT, padx=5)

def upload_images():
    global uploaded_images
    uploaded_images = filedialog.askopenfilenames(
        filetypes=[("Images", "*.png *.jpg *.jpeg")]
    )
    show_images()

def show_images():
    for w in image_frame.winfo_children():
        w.destroy()

    for path in uploaded_images:
        img = Image.open(path).resize((150,150))
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(image_frame, image=tk_img, bg="black")
        lbl.image = tk_img
        lbl.pack(side=tk.LEFT, padx=10)

def predict_images():
    y_true.clear()
    y_pred.clear()
    confidences.clear()

    for w in result_frame.winfo_children():
        w.destroy()

    for path in uploaded_images:
        img = image.load_img(path, target_size=IMG_SIZE)
        arr = image.img_to_array(img)/255.0
        arr = np.expand_dims(arr, axis=0)

        pred = model.predict(arr)[0]
        idx = np.argmax(pred)

        y_pred.append(idx)
        y_true.append(class_names.index(actual_label.get()))
        confidences.append(pred[idx]*100)

        label = class_names[idx]
        color = "green" if label=="forest" else "red"

        tk.Label(result_frame,
                 text=f"{os.path.basename(path)} → {label.upper()} ({pred[idx]*100:.2f}%)",
                 fg=color, bg="black",
                 font=("Arial", 12, "bold")).pack(anchor="w")

def show_analysis():
    for w in graph_frame.winfo_children():
        w.destroy()

    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1,3, figsize=(14,4))

    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    axes[1].bar(range(len(confidences)), confidences, color="orange")
    axes[1].set_title("Prediction Confidence")
    axes[1].set_xlabel("Image Index")
    axes[1].set_ylabel("Confidence (%)")

    unique, counts = np.unique(y_pred, return_counts=True)
    axes[2].bar([class_names[i] for i in unique], counts,
                color=["red","green"])
    axes[2].set_title("Predicted Class Distribution")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Count")

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

tk.Button(control_frame, text="Upload Images",
          width=15, command=upload_images).pack(side=tk.LEFT, padx=10)

tk.Button(control_frame, text="Predict",
          width=15, command=predict_images).pack(side=tk.LEFT, padx=10)

tk.Button(control_frame, text="Show Analysis",
          width=20, command=show_analysis).pack(side=tk.LEFT, padx=10)

# ---------- DISPLAY ----------
image_frame = tk.Frame(root, bg="black")
image_frame.pack(pady=15)

result_frame = tk.Frame(root, bg="black")
result_frame.pack(pady=10)

graph_frame = tk.Frame(root, bg="black")
graph_frame.pack(pady=20)

root.mainloop()
