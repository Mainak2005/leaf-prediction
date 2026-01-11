from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import json
import io
import os


app = FastAPI(title="Potato Leaf Disease Classifier")


MODEL_PATH = r"C:\Users\MAINAK\Desktop\execute\Potato Leaf Disease\potato_leaf_cnn.h5"
LABEL_PATH = r"C:\Users\MAINAK\Desktop\execute\Potato Leaf Disease\class_labels.json"

model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r") as f:
    class_labels = json.load(f)

idx_to_class = {v: k for k, v in class_labels.items()}


def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def hello():
    return {"message": " Potato Leaf Disease Prediction"}


@app.get("/about")
def about():
    return {"message": "Prediction model for getting the healthy or diseased potato leaf"}




@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img_array = preprocess_image(img_bytes)
        preds = model.predict(img_array)
        class_idx = np.argmax(preds, axis=1)[0]
        class_name = idx_to_class[class_idx]
        confidence = float(np.max(preds))
        return JSONResponse({
            "class": class_name,
            "confidence": f"{confidence*100:.2f}%"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
