"""
Prediction Module
"""

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from src.feature_extraction import extract_features
from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)

# Load trained model
model = load_model(MODEL_PATH)

# Load scaler
scaler = joblib.load(SCALER_PATH)

# Load label encoder
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def predict_emotion(audio_file):
    """
    Predict emotion from an audio file.
    """

    # Extract MFCC features
    features = extract_features(audio_file)

    if features is None:
        return None

    # Reshape
    features = np.array(features).reshape(1, -1)

    # Scale
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features, verbose=0)

    predicted_class = np.argmax(prediction)

    emotion = label_encoder.inverse_transform([predicted_class])[0]

    confidence = float(np.max(prediction)) * 100

    return {
        "emotion": emotion,
        "confidence": round(confidence, 2)
    }


if __name__ == "__main__":

    audio_path = input("Enter audio file path: ")

    result = predict_emotion(audio_path)

    if result:

        print("\nPrediction")
        print("-------------------------")
        print("Emotion   :", result["emotion"])
        print("Confidence:", result["confidence"], "%")