import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define image size and batch size
IMG_SIZE = (512, 512)  # Changed to 512x512
BATCH_SIZE = 16

# Set dataset directories
train_dir = "pneumonia detection/chest_xray/train"
test_dir = "pneumonia detection/chest_xray/test"

# Load dataset using image_dataset_from_directory
train_dataset = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Define ANN Model
def build_ann():
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(512, 512, 3)),  # Updated input shape
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define CNN Model
def build_cnn():
    model = keras.Sequential([
        layers.Rescaling(1./255, input_shape=(512, 512, 3)),  # Updated input shape
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and Evaluate ANN
ann_model = build_ann()
history_ann = ann_model.fit(train_dataset, epochs=10, validation_data=test_dataset)
ann_model.save("pneumonia_prediction_ann.h5")

# Train and Evaluate CNN
cnn_model = build_cnn()
history_cnn = cnn_model.fit(train_dataset, epochs=10, validation_data=test_dataset)
cnn_model.save("pneumonia_prediction_cnn.h5")

# Extract features using CNN
feature_extractor = keras.applications.MobileNetV2(include_top=False, input_shape=(512, 512, 3), pooling='avg')  # Updated input shape
X, y = [], []
for images, labels in tqdm(train_dataset, desc="Extracting features"):
    features = feature_extractor.predict(images)
    X.extend(features)
    y.extend(labels.numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')

# Plot Accuracy and Loss for CNN
def plot_history(history, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'{title} - Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{title} - Loss')
    plt.show()

plot_history(history_ann, "ANN")
plot_history(history_cnn, "CNN")

# Save ANN Model
ann_model.save("ann_model_pnrumonia_detectio.h5")

# Save CNN Model
cnn_model.save("cnn_model_pnrumonia_detection.h5")

# Save Random Forest Model using joblib
import joblib
joblib.dump(rf_model, "random_forest_pnrumonia_detectio.pkl")