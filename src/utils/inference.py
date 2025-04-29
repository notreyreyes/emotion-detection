import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from typing import Tuple, List, Union, Optional

def load_image(image_path: str, grayscale: bool = False, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        grayscale: Whether to load the image in grayscale
        target_size: Target size for resizing the image
        
    Returns:
        Preprocessed image array
    """
    try:
        color_mode = "grayscale" if grayscale else "rgb"
        pil_image = image.load_img(image_path, color_mode=color_mode, target_size=target_size)
        return image.img_to_array(pil_image)
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")

def load_detection_model(model_path: str) -> cv2.CascadeClassifier:
    """Load the face detection model.
    
    Args:
        model_path: Path to the Haar Cascade model file
        
    Returns:
        Loaded Cascade Classifier
    """
    try:
        detection_model = cv2.CascadeClassifier(model_path)
        if detection_model.empty():
            raise ValueError(f"Failed to load detection model from {model_path}")
        return detection_model
    except Exception as e:
        raise ValueError(f"Error loading detection model: {str(e)}")

def detect_faces(detection_model: cv2.CascadeClassifier, gray_image_array: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detect faces in a grayscale image.
    
    Args:
        detection_model: Loaded Cascade Classifier
        gray_image_array: Grayscale image array
        
    Returns:
        List of detected face coordinates (x, y, width, height)
    """
    try:
        faces = detection_model.detectMultiScale(
            gray_image_array,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
    except Exception as e:
        print(f"Error detecting faces: {str(e)}")
        return []

def draw_bounding_box(face_coordinates: Tuple[int, int, int, int], image_array: np.ndarray, color: Union[List[int], Tuple[int, int, int]]) -> None:
    """Draw a bounding box around a detected face.
    
    Args:
        face_coordinates: Face coordinates (x, y, width, height)
        image_array: Image array to draw on
        color: RGB color tuple or list
    """
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates: Tuple[int, int, int, int], offsets: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Apply offsets to face coordinates.
    
    Args:
        face_coordinates: Face coordinates (x, y, width, height)
        offsets: Tuple of x and y offsets
        
    Returns:
        Adjusted coordinates (x1, x2, y1, y2)
    """
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates: Tuple[int, int, int, int], image_array: np.ndarray, text: str,
             color: Union[List[int], Tuple[int, int, int]], x_offset: int = 0, y_offset: int = 0,
             font_scale: float = 2, thickness: int = 2) -> None:
    """Draw text on the image.
    
    Args:
        coordinates: Face coordinates (x, y, width, height)
        image_array: Image array to draw on
        text: Text to display
        color: RGB color tuple or list
        x_offset: X offset for text position
        y_offset: Y offset for text position
        font_scale: Font scale factor
        thickness: Text thickness
    """
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def get_colors(num_classes: int) -> np.ndarray:
    """Generate a color palette for visualization.
    
    Args:
        num_classes: Number of colors to generate
        
    Returns:
        Array of RGB colors
    """
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    return (np.asarray(colors) * 255).astype(np.uint8)

