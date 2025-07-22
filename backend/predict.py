# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

model = load_model("mask_detector.h5")

labels_dict = {0: 'No Mask 😷❌', 1: 'Mask On 😷✅'}

def predict_mask(image_bytes):
    # Convert byte data to image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize((150, 150))
    image = np.array(image)

    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]

    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0][0]
    label = 1 if pred > 0.5 else 0

    return labels_dict[label]
