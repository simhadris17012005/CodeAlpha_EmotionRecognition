"""
Streamlit App
Speech Emotion Recognition
"""

import os
import tempfile
import streamlit as st

from src.predict import predict_emotion

st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Speech Emotion Recognition")
st.write("Upload a WAV audio file to predict the emotion.")

uploaded_file = st.file_uploader(
    "Choose a WAV file",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Emotion"):

        with st.spinner("Analyzing Audio..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            result = predict_emotion(temp_path)

            os.remove(temp_path)

        if result:

            st.success("Prediction Completed!")

            st.subheader("Prediction Result")

            st.write(f"**Emotion:** {result['emotion']}")

            st.write(f"**Confidence:** {result['confidence']} %")

        else:

            st.error("Could not process the uploaded audio.")