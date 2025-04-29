import numpy as np
from typing import Union, List, Tuple
from tensorflow.keras.utils import to_categorical as keras_to_categorical


def preprocess_input(x: np.ndarray, v2: bool = True) -> np.ndarray:
    """Preprocess input data for the model.
    
    Args:
        x: Input data array
        v2: Whether to use version 2 preprocessing (normalize to [-1, 1])
        
    Returns:
        Preprocessed array
    """
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x


def load_and_resize_image(image_path: str, target_size: Union[Tuple[int, int], List[int]]) -> np.ndarray:
    """Load and resize an image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (width, height)
        
    Returns:
        Resized image array
    """
    try:
        from PIL import Image
        img = Image.open(image_path)
        img = img.resize(target_size)
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {str(e)}")


def to_categorical(integer_classes: Union[np.ndarray, List[int]], num_classes: int = 2) -> np.ndarray:
    """Convert class vector to binary class matrix.
    
    Args:
        integer_classes: Array of class indices
        num_classes: Total number of classes
        
    Returns:
        Binary matrix representation of the input
    """
    return keras_to_categorical(integer_classes, num_classes)
