# 🎤 Emotion Recognition from Speech

A Machine Learning & Deep Learning project developed as part of the **CodeAlpha Machine Learning Internship**.

## 📌 Project Overview

This project recognizes human emotions from speech audio using **Mel-Frequency Cepstral Coefficients (MFCC)** and a **Deep Neural Network (TensorFlow/Keras)**.

Users can upload a `.wav` audio file through a Streamlit web application, and the trained model predicts the speaker's emotion.

---

## 🎯 Objectives

* Recognize emotions from speech audio.
* Extract MFCC features from `.wav` files.
* Train a Deep Learning model for emotion classification.
* Build an interactive Streamlit web application.
* Save and reuse the trained model for predictions.

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Librosa
* NumPy
* Pandas
* Matplotlib
* Joblib
* Streamlit

---

## 📂 Project Structure

```text
CodeAlpha_EmotionRecognition/
│
├── dataset/
│   └── RAVDESS/
│
├── models/
│   ├── emotion_model.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── notebook.ipynb
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 📊 Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

Supported emotions:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fear
* Disgust
* Surprised

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/simhadris17012005/CodeAlpha_EmotionRecognition.git
```

Move into the project folder:

```bash
cd CodeAlpha_EmotionRecognition
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python -m src.train
```

This creates:

* emotion_model.keras
* scaler.pkl
* label_encoder.pkl

inside the `models` folder.

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

Open the Streamlit URL in your browser, upload a `.wav` audio file, and view the predicted emotion with its confidence score.

---

## 📈 Features

* Speech Emotion Recognition
* MFCC Feature Extraction
* Deep Learning Model
* Feature Scaling
* Label Encoding
* Model Evaluation
* Confusion Matrix
* Streamlit User Interface

---

## 📷 Output

The application allows users to:

* Upload a speech audio file
* Predict the speaker's emotion
* Display prediction confidence

---

## 👨‍💻 Author

**Simhadri**

Machine Learning Intern — CodeAlpha

---
## Dataset

This project uses the RAVDESS dataset.

Download:
https://zenodo.org/record/1188976

After downloading, place it inside:

dataset/RAVDESS/

## 📄 License

This project is licensed under the MIT License.
