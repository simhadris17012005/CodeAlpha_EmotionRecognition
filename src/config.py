"""
Configuration File
"""

import os

# ===========================
# Project Paths
# ===========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "RAVDESS")

MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_model.keras")

SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")


# ===========================
# Audio Parameters
# ===========================

SAMPLE_RATE = 22050

N_MFCC = 40

DURATION = 3

OFFSET = 0.5


# ===========================
# Model Parameters
# ===========================

TEST_SIZE = 0.2

RANDOM_STATE = 42

EPOCHS = 50

BATCH_SIZE = 32