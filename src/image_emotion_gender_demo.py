import os
import sys
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from typing import Tuple, List

from utils.datasets import get_labels, get_class_to_arg
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model, load_image
from utils.preprocessor import preprocess_input

def process_image(image_path: str) -> None:
    """Process an image to detect faces, emotions, and gender.
    
    Args:
        image_path: Path to the input image
    """
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # parameters for loading data and images
    detection_model_path = os.path.join(project_root, 'trained_models', 'detection_models', 'haarcascade_frontalface_default.xml')
    emotion_model_path = os.path.join(project_root, 'trained_models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
    gender_model_path = os.path.join(project_root, 'trained_models', 'gender_models', 'gender_mini_XCEPTION.21-0.95.hdf5')
    
    # Get labels and label mappings
    emotion_labels = get_labels('fer2013')
    gender_labels = get_labels('imdb')
    gender_class_to_arg = get_class_to_arg('imdb')
    
    # hyper-parameters for bounding boxes shape
    gender_offsets = (30, 60)
    emotion_offsets = (20, 40)

    try:
        # loading models
        face_detection = load_detection_model(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        gender_classifier = load_model(gender_model_path, compile=False)

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]
        gender_target_size = gender_classifier.input_shape[1:3]

        # loading images
        rgb_image = load_image(image_path, grayscale=False)
        gray_image = load_image(image_path, grayscale=True)
        gray_image = np.squeeze(gray_image)
        gray_image = gray_image.astype('uint8')

        faces = detect_faces(face_detection, gray_image)
        if len(faces) == 0:
            print("No faces detected in the image")
            return

        for face_coordinates in faces:
            # Process gender
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            
            # Add boundary checks
            x1 = max(0, x1)
            x2 = min(rgb_image.shape[1], x2)
            y1 = max(0, y1)
            y2 = min(rgb_image.shape[0], y2)
            
            # Check if the face region is valid
            if x2 <= x1 or y2 <= y1:
                print("Invalid face region detected, skipping...")
                continue
            
            rgb_face = rgb_image[y1:y2, x1:x2]
            
            if rgb_face.size == 0:
                print("Empty face region detected, skipping...")
                continue

            # Process emotion
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            
            # Add boundary checks
            x1 = max(0, x1)
            x2 = min(gray_image.shape[1], x2)
            y1 = max(0, y1)
            y2 = min(gray_image.shape[0], y2)
            
            # Check if the face region is valid
            if x2 <= x1 or y2 <= y1:
                print("Invalid face region detected, skipping...")
                continue
            
            gray_face = gray_image[y1:y2, x1:x2]
            
            if gray_face.size == 0:
                print("Empty face region detected, skipping...")
                continue

            try:
                rgb_face = cv2.resize(rgb_face, gender_target_size)
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except Exception as e:
                print(f"Error resizing face: {e}")
                continue

            # Predict gender (using grayscale)
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

            # Set color based on gender (blue for women, red for men)
            is_woman = gender_label_arg == gender_class_to_arg['woman']
            color = (0, 0, 255) if is_woman else (255, 0, 0)

            # Draw results
            draw_bounding_box(face_coordinates, rgb_image, color)
            draw_text(face_coordinates, rgb_image, gender_text, color, 0, -20, 1, 1)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -45, 1, 1)

        # Save the result
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(project_root, 'images', 'predicted_test_image.png')
        cv2.imwrite(output_path, bgr_image)
        print(f"Processed image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python image_emotion_gender_demo.py <image_path>")
        sys.exit(1)
    
    process_image(sys.argv[1])
