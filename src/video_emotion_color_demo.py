#PAPER CODE REFRACTORED 

import os
from statistics import mode
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import platform

from utils.datasets import get_labels
from utils.inference import detect_faces, draw_text, draw_bounding_box, apply_offsets, load_detection_model
from utils.preprocessor import preprocess_input

def initialize_camera():
    """Initialize the camera with appropriate settings for the platform."""
    if platform.system() == 'Darwin':  # macOS
        # Try different camera indices
        for i in range(3):  # Try first 3 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Set some basic properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                return cap
        raise RuntimeError("Could not open any camera. Please check camera permissions in System Preferences > Security & Privacy > Camera")
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        return cap

def main():
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # parameters for loading data and images
    detection_model_path = os.path.join(project_root, 'trained_models', 'detection_models', 'haarcascade_frontalface_default.xml')
    emotion_model_path = os.path.join(project_root, 'trained_models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    try:
        # loading models
        face_detection = load_detection_model(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []

        # Initialize camera
        print("Initializing camera...")
        video_capture = initialize_camera()
        print("Camera initialized successfully!")

        # starting video streaming
        cv2.namedWindow('window_frame')
        
        print("Press 'q' to quit")
        while True:
            ret, bgr_image = video_capture.read()
            if not ret:
                print("Error: Could not read frame from video capture")
                break
                
            gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            faces = detect_faces(face_detection, gray_image)
            
            if len(faces) == 0:
                print("No faces detected in the current frame")
            else:
                print(f"Detected {len(faces)} face(s)")

            for face_coordinates in faces:
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
                    gray_face = cv2.resize(gray_face, emotion_target_size)
                except Exception as e:
                    print(f"Error resizing face: {e}")
                    continue

                gray_face = preprocess_input(gray_face, True)
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                
                try:
                    emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
                    emotion_probability = np.max(emotion_prediction)
                    emotion_label_arg = np.argmax(emotion_prediction)
                    emotion_text = emotion_labels[emotion_label_arg]
                    emotion_window.append(emotion_text)

                    if len(emotion_window) > frame_window:
                        emotion_window.pop(0)
                    try:
                        emotion_mode = mode(emotion_window)
                    except:
                        continue

                    # Define colors for different emotions
                    color_map = {
                        'angry': (255, 0, 0),
                        'sad': (0, 0, 255),
                        'happy': (255, 255, 0),
                        'surprise': (0, 255, 255),
                        'neutral': (0, 255, 0)
                    }
                    
                    color = np.array(color_map.get(emotion_text, (0, 255, 0))) * emotion_probability
                    color = color.astype(int).tolist()

                    draw_bounding_box(face_coordinates, rgb_image, color)
                    draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
                    
                except Exception as e:
                    print(f"Error in emotion prediction: {e}")
                    continue

            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('window_frame', bgr_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
