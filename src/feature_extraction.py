"""
Feature Extraction Module
Extract MFCC Features from Audio Files
"""

import librosa
import numpy as np

from src.config import (
    SAMPLE_RATE,
    N_MFCC,
    DURATION,
    OFFSET
)


def extract_features(file_path):
    """
    Extract MFCC features from an audio file.

    Parameters
    ----------
    file_path : str
        Path to .wav file

    Returns
    -------
    numpy.ndarray
        MFCC feature vector
    """

    try:

        # Load Audio
        signal, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            offset=OFFSET
        )

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Mean of MFCC
        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:

        print(f"Error processing {file_path}")
        print(e)

        return None


if __name__ == "__main__":

    print("Feature Extraction Module Loaded Successfully!")