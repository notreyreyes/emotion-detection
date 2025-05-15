from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model
from typing import Dict, Any, List
import logging

from src.utils.datasets import get_labels, get_class_to_arg
from src.utils.inference import detect_faces, apply_offsets, load_detection_model
from src.utils.preprocessor import preprocess_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Emotion Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize models
def load_models():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        #model paths
        detection_model_path = os.path.join(script_dir, 'trained_models', 'detection_models', 'haarcascade_frontalface_default.xml')
        emotion_model_path = os.path.join(script_dir, 'trained_models', 'emotion_models', 'fer2013_mini_XCEPTION.102-0.66.hdf5')
        gender_model_path = os.path.join(script_dir, 'trained_models', 'gender_models', 'gender_mini_XCEPTION.21-0.95.hdf5')
        
        #load models
        face_detection = load_detection_model(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        gender_classifier = load_model(gender_model_path, compile=False)
        
        logger.info("Models loaded successfully")
        return face_detection, emotion_classifier, gender_classifier
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading models")

try:
    face_detection, emotion_classifier, gender_classifier = load_models()
except Exception as e:
    logger.error(f"Failed to initialize models: {str(e)}")
    raise

def process_image(image: Image.Image) -> Dict[str, Any]:
    try:
        # Get labels
        emotion_labels = get_labels('fer2013')
        gender_labels = get_labels('imdb')
        
        # Set offsets
        gender_offsets = (30, 60)
        emotion_offsets = (20, 40)
        
        # Get target sizes
        emotion_target_size = emotion_classifier.input_shape[1:3]
        gender_target_size = gender_classifier.input_shape[1:3]
        
        # Convert image to RGB and grayscale
        rgb_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        #Detect faces
        faces = detect_faces(face_detection, gray_image)
        
        if len(faces) == 0:
            return {"error": "No faces detected in the image"}
        
        results = {
            "emotions": {},
            "gender": None,
            "gender_confidence": None,
            "face_count": len(faces)
        }
        
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

            try:
                rgb_face = cv2.resize(rgb_face, gender_target_size)
                gray_face = cv2.resize(gray_image[y1:y2, x1:x2], emotion_target_size)
            except Exception as e:
                logger.error(f"Error resizing face: {str(e)}")
                continue

            # Predict gender
            gender_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
            gender_face = preprocess_input(gender_face, True)
            gender_face = np.expand_dims(gender_face, 0)
            gender_face = np.expand_dims(gender_face, -1)
            gender_prediction = gender_classifier.predict(gender_face, verbose=0)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]
            gender_confidence = float(gender_prediction[0][gender_label_arg])

            # Predict emotion
            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face, verbose=0)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_confidence = float(emotion_prediction[0][emotion_label_arg])

            # Store results
            results["gender"] = gender_text
            results["gender_confidence"] = gender_confidence
            results["emotions"][emotion_text] = emotion_confidence
        
        return results
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": f"Error processing image: {str(e)}"}

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the uploaded file
        contents = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        try:
            
            image = Image.open(temp_file_path)
            results = process_image(image)
            
           
            os.unlink(temp_file_path)
            
            return JSONResponse(content=results)
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    except Exception as e:
        logger.error(f"Error in analyze_image endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        
        contents = await file.read()
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        try:
            
            cap = cv2.VideoCapture(temp_file_path)
            results = []
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                if frame_count % 5 != 0:
                    continue
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                
                
                frame_results = process_image(image)
                if "error" not in frame_results:
                    results.append(frame_results)
            
            cap.release()
            os.unlink(temp_file_path)
            
            if results:
                aggregated_results = {
                    "emotions": {},
                    "gender": None,
                    "gender_confidence": None,
                    "face_count": 0
                }
                
                
                gender_counts = {}
                for result in results:
                    if result["gender"]:
                        gender_counts[result["gender"]] = gender_counts.get(result["gender"], 0) + 1
                    for emotion, confidence in result["emotions"].items():
                        aggregated_results["emotions"][emotion] = aggregated_results["emotions"].get(emotion, 0) + confidence
                    aggregated_results["face_count"] += result["face_count"]
                
                if aggregated_results["emotions"]:
                    for emotion in aggregated_results["emotions"]:
                        aggregated_results["emotions"][emotion] /= len(results)
                
                if gender_counts:
                    aggregated_results["gender"] = max(gender_counts, key=gender_counts.get)
                    aggregated_results["gender_confidence"] = gender_counts[aggregated_results["gender"]] / len(results)
                
                aggregated_results["face_count"] = aggregated_results["face_count"] // len(results)
                
                return JSONResponse(content=aggregated_results)
            else:
                return JSONResponse(content={"error": "No faces detected in the video"})
        except Exception as e:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    except Exception as e:
        logger.error(f"Error in analyze_video endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000) 