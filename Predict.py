import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/deforestation_model.h5")

class_names = ["deforested", "forest"]

test_dir = "test_images"

for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    confidence = np.max(prediction) * 100
    predicted_class = class_names[np.argmax(prediction)]

    print(f"{img_name} → {predicted_class} ({confidence:.2f}%)")
