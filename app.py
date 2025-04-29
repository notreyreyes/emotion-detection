import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model

from utils.datasets import get_labels, get_class_to_arg
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model, load_image
from utils.preprocessor import preprocess_input

# Set page config
st.set_page_config(
    page_title="Emotion & Gender Detection",
    page_icon="😊",
    layout="wide"
)

# Title and description
st.title("Emotion & Gender Detection App")
st.write("Upload an image or use your webcam to detect faces, emotions, and gender!")

# Initialize models (we'll do this once and cache it)
@st.cache_resource
def load_models():
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Load model paths
    detection_model_path = os.path.join(project_root, 'trained_models', 'detection_models', 'haarcascade_frontalface_default.xml')
    emotion_model_path = os.path.join(project_root, 'trained_models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
    gender_model_path = os.path.join(project_root, 'trained_models', 'gender_models', 'gender_mini_XCEPTION.21-0.95.hdf5')
    
    # Load models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)
    gender_classifier = load_model(gender_model_path, compile=False)
    
    return face_detection, emotion_classifier, gender_classifier

# Process image function
def process_image(image, face_detection, emotion_classifier, gender_classifier):
    # Get labels
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    gender_class_to_arg = get_class_to_arg('imdb')
    
    # Set offsets
    gender_offsets = (30, 60)
    emotion_offsets = (20, 40)
    
    # Get target sizes
    emotion_target_size = emotion_classifier.input_shape[1:3]
    gender_target_size = gender_classifier.input_shape[1:3]
    
    # Convert image to RGB and grayscale
    rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detect_faces(face_detection, gray_image)
    
    if len(faces) == 0:
        st.warning("No faces detected in the image!")
        return image
    
    # Process each face
    for face_coordinates in faces:
        # Process gender
        x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
        x1 = max(0, x1)
        x2 = min(rgb_image.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(rgb_image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        rgb_face = rgb_image[y1:y2, x1:x2]
        
        if rgb_face.size == 0:
            continue

        # Process emotion
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        x1 = max(0, x1)
        x2 = min(gray_image.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(gray_image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        gray_face = gray_image[y1:y2, x1:x2]
        
        if gray_face.size == 0:
            continue

        try:
            rgb_face = cv2.resize(rgb_face, gender_target_size)
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except:
            continue

        # Predict gender
        gender_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
        gender_face = preprocess_input(gender_face, True)
        gender_face = np.expand_dims(gender_face, 0)
        gender_face = np.expand_dims(gender_face, -1)
        gender_prediction = gender_classifier.predict(gender_face, verbose=0)
        gender_label_arg = np.argmax(gender_prediction)
        gender_text = gender_labels[gender_label_arg]

        # Predict emotion
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

        # Set color based on gender
        is_woman = gender_label_arg == gender_class_to_arg['woman']
        color = (0, 0, 255) if is_woman else (255, 0, 0)

        # Draw results
        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 1)
        draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -45, 1, 1)
    
    return Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))

# Main app
def main():
    # Load models
    face_detection, emotion_classifier, gender_classifier = load_models()
    
    # Create two columns for input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Image", use_column_width=True)
            
            if st.button("Process Uploaded Image"):
                with st.spinner("Processing..."):
                    result = process_image(image, face_detection, emotion_classifier, gender_classifier)
                    st.image(result, caption="Processed Image", use_column_width=True)
    
    with col2:
        st.header("Webcam")
        st.write("Click the button below to start your webcam")
        
        if st.button("Start Webcam"):
            # Use streamlit's camera input
            img_file_buffer = st.camera_input("Take a picture")
            
            if img_file_buffer is not None:
                image = Image.open(img_file_buffer)
                with st.spinner("Processing..."):
                    result = process_image(image, face_detection, emotion_classifier, gender_classifier)
                    st.image(result, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main() 