import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

MODEL_PATH = "model/deforestation_model.h5"
TEST_DIR = "test_images"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)
class_names = ["deforested", "forest"]

root = tk.Tk()
root.title("Deforestation Detection Dashboard")
root.geometry("1200x700")
root.configure(bg="black")

# ---------- TITLE ----------
title = tk.Label(root, text="Deforestation Detection System",
                 font=("Arial", 22, "bold"),
                 fg="white", bg="black")
title.pack(pady=10)

# ---------- BUTTON FRAME ----------
btn_frame = tk.Frame(root, bg="black")
btn_frame.pack()

def show_confusion_matrix():
    test_ds = image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=32,
        shuffle=False
    )

    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = np.argmax(model.predict(test_ds), axis=1)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues", ax=ax)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Test Dataset)")

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

tk.Button(btn_frame, text="Show Confusion Matrix",
          command=show_confusion_matrix,
          width=22).pack(side=tk.LEFT, padx=10)

tk.Button(btn_frame, text="Exit",
          command=root.destroy,
          width=10).pack(side=tk.LEFT)

# ---------- GRAPH AREA ----------
graph_frame = tk.Frame(root, bg="black")
graph_frame.pack(pady=20)

root.mainloop()
