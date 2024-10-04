from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import tempfile

# Step 1: Define a function to preprocess input images
def preprocess_image(img_path, img_height=500, img_width=500):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Step 2: Load the trained model
model = load_model('breast_cancer_cnn_model.h5')

# Map class indices to class labels
class_names = ['benign', 'malignant', 'normal']  # Adjust these as per your training labels

# Initialize FastAPI
app = FastAPI()

# Step 3: Define a FastAPI endpoint for image predictions
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp.flush()
        tmp.seek(0)

        processed_image = preprocess_image(tmp.name)
        predictions = model.predict(processed_image)

        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_label = class_names[predicted_class_index]

        return JSONResponse(content={"predicted_class_index": predicted_class_index, "predicted_class_label": predicted_class_label})

