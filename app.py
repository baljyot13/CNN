import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib
import cv2
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Initialize Flask app
app = Flask(__name__)

# Load trained models
ann_model = keras.models.load_model("models/ann_model_pnrumonia_detectio.h5")
cnn_model = keras.models.load_model("models/cnn_model_pnrumonia_detection.h5")
rf_model = joblib.load("models/random_forest_pnrumonia_detectio.pkl")

# Load feature extractor for Random Forest
feature_extractor = keras.applications.MobileNetV2(include_top=False, input_shape=(512, 512, 3), pooling='avg')

# Define image preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (512, 512))  # Resize to match training size
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to extract features for Random Forest
def extract_features(image):
    features = feature_extractor.predict(image)
    return features

# Function to get model predictions
def predict_pneumonia(image_path):
    img = preprocess_image(image_path)

    # ANN Prediction
    ann_pred = ann_model.predict(img)[0][0]
    ann_result = "Pneumonia Detected" if ann_pred > 0.5 else "Normal"

    # CNN Prediction
    cnn_pred = cnn_model.predict(img)[0][0]
    cnn_result = "Pneumonia Detected" if cnn_pred > 0.5 else "Normal"

    # Random Forest Prediction
    features = extract_features(img)
    rf_pred = rf_model.predict(features)[0]
    rf_result = "Pneumonia Detected" if rf_pred < 0.5 else "Normal"

    return {
        "ANN Prediction": ann_result,
        "ANN Confidence": f"{ann_pred * 100:.2f}%",
        "CNN Prediction": cnn_result,
        "CNN Confidence": f"{cnn_pred * 100:.2f}%",
        "Random Forest Prediction": rf_result
    }

# Define upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file!"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Get predictions
            results = predict_pneumonia(file_path)

            # Render results
            return render_template("result.html", results=results, image_path=file_path)

    return render_template("index.html")

# Run Flask App
if __name__ == "__main__":
    app.run(port=5004,debug=True)
