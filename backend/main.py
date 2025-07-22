from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the model
model = load_model("backend/mask_detector.h5")


# Label dictionary
labels_dict = {0: 'No Mask 😷❌', 1: 'Mask On 😷✅'}

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model
class ImageData(BaseModel):
    image: str  # base64 string

@app.get("/")
def read_root():
    return {"message": "Mask Detector is Live"}

@app.post("/predict")
async def predict(data: ImageData):
    try:
        # Decode base64 to bytes
        img_bytes = base64.b64decode(data.image.split(',')[1])

        # Convert bytes to numpy array
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize and normalize
        img = cv2.resize(img, (150, 150))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        pred = model.predict(img)[0][0]
        label = 1 if pred > 0.5 else 0
        result = labels_dict[label]
        return {"prediction": result}

    except Exception as e:
        return {"error": str(e)}
