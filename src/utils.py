"""
Utility Functions
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)


def evaluate_model(y_true, y_pred, label_encoder):
    """
    Print evaluation metrics.
    """

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def plot_history(history):
    """
    Plot training history.
    """

    # Accuracy
    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()

    # Loss
    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()