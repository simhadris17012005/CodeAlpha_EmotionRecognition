"""
Model Training Module
"""

import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from src.preprocessing import preprocess_data
from src.config import (
    MODEL_PATH,
    EPOCHS,
    BATCH_SIZE
)
from src.utils import evaluate_model, plot_history


def build_model(input_shape, num_classes):
    """
    Build Deep Neural Network
    """

    model = Sequential([

        Dense(256, activation="relu", input_shape=(input_shape,)),
        Dropout(0.30),

        Dense(128, activation="relu"),
        Dropout(0.30),

        Dense(64, activation="relu"),
        Dropout(0.20),

        Dense(num_classes, activation="softmax")

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():

    print("=" * 60)
    print("Speech Emotion Recognition Model Training")
    print("=" * 60)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        scaler
    ) = preprocess_data()

    print("\nDataset Loaded Successfully!")
    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    model = build_model(
        X_train.shape[1],
        len(label_encoder.classes_)
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("\nTraining Started...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    print("\nTraining Completed!")

    print("\nEvaluating Model...")

    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    evaluate_model(
        y_test,
        predicted_classes,
        label_encoder
    )

    plot_history(history)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    model.save(MODEL_PATH)

    print("\nModel Saved Successfully!")
    print(f"Location : {MODEL_PATH}")


if __name__ == "__main__":

    train()