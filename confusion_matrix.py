import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TEST_DIR = "test_images"

if not os.path.exists(TEST_DIR):
    raise SystemExit("❌ test_images folder not found")

if not any(os.scandir(TEST_DIR)):
    raise SystemExit("❌ test_images is empty")

model = tf.keras.models.load_model("model/deforestation_model.h5")

img_size = 224
batch_size = 16

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

if test_data.samples == 0:
    raise SystemExit(
        "❌ No images found.\n"
        "Ensure structure:\n"
        "test_images/forest/*.jpg\n"
        "test_images/deforested/*.jpg"
    )

predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

labels = list(test_data.class_indices.keys())
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Dataset)")
plt.tight_layout()
plt.show()

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))
