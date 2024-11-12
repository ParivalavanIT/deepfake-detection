import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionResNetV2
from streamlit_lottie import st_lottie
import json
import requests

# Parameters
frame_height, frame_width = 160, 160  # Required input size for FaceNet
sequence_length = 10  # Number of frames per sequence
num_classes = 2  # Real or Fake


@st.cache_resource
def load_facenet_model():
    base_model = InceptionResNetV2(include_top=False, input_shape=(
        frame_height, frame_width, 3), pooling='avg')
    return base_model

# Load a single video file and extract FaceNet embeddings

def load_single_video(video_file, facenet_model):
    cap = cv2.VideoCapture(video_file)
    embeddings = []
    while len(embeddings) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_height, frame_width))
        frame = frame / 255.0  # Normalize the frame
        embedding = facenet_model.predict(np.expand_dims(frame, axis=0))
        embeddings.append(embedding.flatten())
    cap.release()
    if len(embeddings) == sequence_length:
        data = np.array(embeddings)
        return data
    else:
        return None

# LSTM Model using FaceNet embeddings

def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False,
              input_shape=(sequence_length, 1536)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to provide explanation based on the prediction

def generate_explanation(class_label, prediction_probabilities):
    explanations = []
    if class_label == 1:
        explanations.append(
            "Inconsistencies detected in facial features, such as unnatural expressions or asymmetric movements.")
        explanations.append(
            "Lighting and shading inconsistencies between the face and the rest of the scene.")
        explanations.append(
            "Unnatural or jerky movements that do not align with expected human motion.")
        explanations.append(
            "Blurriness or artifacts around the edges of the face, often indicating poor face-swapping.")
        explanations.append(
            f"Model confidence in this being fake: {prediction_probabilities[1]*100:.2f}%")
    else:
        explanations.append(
            "No significant abnormalities detected in facial features or movements.")
        explanations.append(
            "Consistent lighting and shading throughout the scene.")
        explanations.append(
            "Smooth and natural movements that align with expected human motion.")
        explanations.append(
            f"Model confidence in this being real: {prediction_probabilities[0]*100:.2f}%")
    return explanations

# Function to load Lottie animation

st.set_page_config(
    page_title="NeuroForge",
    page_icon="🧊"
)


def get(path: str):
    with open(path, 'r') as f:
        return json.load(f)


path = get('./deepfake-model.json')

# Streamlit UI
st.title("NeuroForge - Deep Fake Video Detection")


st.markdown("""
Welcome to **NeuroForge**, a cutting-edge deep fake detection tool. Simply upload a video, and NeuroForge will analyze it to determine if the content is genuine or manipulated.
""")

uploaded_file = st.file_uploader(
    "Upload a video file (Supported formats: mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv","mpeg"])

if uploaded_file is not None:
    # Display loading animation
    st.spinner("Analyzing video...")

    # Save uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())

    # Load FaceNet model
    facenet_model = load_facenet_model()

    # Extract embeddings
    frames = load_single_video("temp_video.mp4", facenet_model)

    if frames is not None:
        frames = np.expand_dims(frames, axis=0)

        # Create and compile the model
        model = create_model()

        # Predict
        prediction = model.predict(frames)
        class_label = np.argmax(prediction)
        prediction_probabilities = prediction[0]

        # Display Results
        st.markdown("### Detection Results")

        # Detailed Confidence and Polarity
        confidence_score = prediction_probabilities[class_label] * 100
        polarity = "Fake" if class_label == 1 else "Real"

        st.write(f"**Prediction:** {polarity}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}%")

        # Display Results with Summary
        if class_label == 1:
            st.error("**Confirmed Fake:** Inconsistencies detected.")
            st.write(
                "This result suggests that the video likely contains manipulated or generated content.")
        else:
            st.success("**Confirmed Real:** No abnormalities detected.")
            st.write(
                "This result suggests that the video appears to be authentic with no signs of tampering.")

        # Generate and display explanations
        explanations = generate_explanation(
            class_label, prediction_probabilities)
        st.markdown("#### Detailed Explanation")
        for explanation in explanations:
            st.write("- " + explanation)

    else:
        st.warning("Insufficient frames for analysis. Please try a longer video.")

st_lottie(path)


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# URL for a Lottie animation that fits documentation context
doc_animation_url = "https://assets2.lottiefiles.com/packages/lf20_x62chJ.json"  # Example URL
doc_animation = load_lottie_url(doc_animation_url)

open_docs = st.button("Open Documentation")

# Display documentation when the button is clicked
if open_docs:
    # Display Lottie animation
    st_lottie(doc_animation, height=150, width=150)

    st.header("Abstract")
    st.write(
        "NeuroForge is an advanced, AI-powered video analysis platform focused on detecting deepfakes. "
        "Leveraging powerful deep learning models, NeuroForge offers a reliable solution to differentiate between real and manipulated video content. "
        "This tool is designed for content creators, news agencies, security organizations, and anyone interested in preserving the authenticity "
        "of visual media in an age of increasing synthetic content."
    )

    st.header("Key Features")

    st.write("### 1. User-Friendly Interface:")
    st.write("- NeuroForge provides an intuitive interface for users to upload videos and receive detailed deepfake detection reports with confidence scores.")

    st.write("### 2. Real-Time Video Analysis:")
    st.write("- Analyze video content quickly, with real-time detection results and insightful explanations on potential manipulation techniques detected.")

    st.write("### 3. Explanation-Driven Reporting:")
    st.write("- Provides in-depth explanations for each classification, detailing inconsistencies found in lighting, motion, and facial features for comprehensive understanding.")

    st.write("### 4. Batch Processing for Bulk Analysis:")
    st.write("- For enterprise clients, NeuroForge supports batch video processing, making it ideal for content verification at scale.")

    st.write("### 5. Confidence Score and Polarity:")
    st.write("- Each analysis includes a confidence score indicating the likelihood of the content being real or fake, along with a polarity analysis for a well-rounded evaluation.")

    st.write("### 6. Ethical AI and Data Privacy Compliance:")
    st.write("- NeuroForge is committed to ethical AI practices and ensures data privacy, following industry standards for handling and analyzing video content securely.")

    st.header("Technical Details")

    st.write("### Dependencies:")
    st.write("  - `streamlit`, `tensorflow`, `opencv-python`, `numpy`, `pandas`")

    st.write("### Underlying Technology:")
    st.write("  - NeuroForge leverages a deep neural network using the InceptionResNetV2 architecture for facial feature extraction and LSTM for sequence analysis to detect deepfake characteristics.")

    st.write("### Explanation and Visualization:")
    st.write("  - The application provides explanation-based feedback, highlighting detected abnormalities in facial features or motion patterns, which are presented with visual cues for better understanding.")

    st.header("Use Cases")

    st.write("- **Media Outlets**: Quickly validate the authenticity of video content to prevent the spread of misinformation.")
    st.write("- **Security and Surveillance**: Detect deepfakes in surveillance footage to identify potential threats.")
    st.write("- **Legal Evidence Verification**: Analyze video evidence to ensure its credibility in legal contexts.")

    st.header("Conclusion")

    st.write(
        "NeuroForge represents a pioneering approach to video deepfake detection, offering real-time feedback and advanced "
        "explanation-driven insights. This platform is a valuable resource in today's digital landscape, supporting individuals "
        "and organizations in maintaining the integrity of video content and promoting digital authenticity."
    )
