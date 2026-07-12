"""
Data Preprocessing Module
"""

import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.feature_extraction import extract_features
from src.config import (
    DATASET_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)


# Emotion Mapping
emotion_dict = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprised"
}


def preprocess_data():

    features = []
    labels = []

    print("Loading Dataset...")

    # Loop through Actor folders
    for actor in os.listdir(DATASET_PATH):

        actor_path = os.path.join(DATASET_PATH, actor)

        if not os.path.isdir(actor_path):
            continue

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                file_path = os.path.join(actor_path, file)

                # Emotion Code
                emotion_code = file.split("-")[2]

                emotion = emotion_dict.get(emotion_code)

                if emotion is None:
                    continue

                # Extract MFCC
                feature = extract_features(file_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)

    X = np.array(features)
    y = np.array(labels)

    print(f"Total Samples : {len(X)}")

    # Encode Labels
    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)

    # Scale Features
    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    # Save Encoder & Scaler
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Preprocessing Completed Successfully!")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        scaler
    )


if __name__ == "__main__":

    preprocess_data()