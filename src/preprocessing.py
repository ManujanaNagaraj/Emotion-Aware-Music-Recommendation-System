import cv2
import numpy as np
from typing import Tuple, Optional

class FacePreprocessor:
    """
    A class to handle face extraction and preprocessing for emotion classification.
    
    This module prepares raw face bounding boxes by cropping, resizing, 
    converting to grayscale, and normalizing pixel values.
    """

    def __init__(self, target_size: Tuple[int, int] = (48, 48)):
        """
        Initializes the FacePreprocessor with a target resize dimension.

        Args:
            target_size (Tuple[int, int]): The dimensions (width, height) to resize faces to.
                Defaults to (48, 48), standard for many emotion models (e.g., FER-2013).
        """
        self.target_size = target_size

    def extract_face(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Crops the face region from the original frame based on the bounding box.

        Args:
            frame (np.ndarray): The original image/frame.
            bbox (Tuple[int, int, int, int]): The bounding box (x, y, w, h).

        Returns:
            Optional[np.ndarray]: The cropped face image, or None if extraction fails.
        """
        if frame is None or bbox is None:
            return None
        
        x, y, w, h = bbox
        
        # Ensure coordinates are within frame boundaries
        h_frame, w_frame = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + w), min(h_frame, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        return frame[y1:y2, x1:x2]

    def to_grayscale(self, face_image: np.ndarray) -> np.ndarray:
        """
        Converts the face image to grayscale.

        Args:
            face_image (np.ndarray): The input face image (BGR).

        Returns:
            np.ndarray: The grayscale image.
        """
        if len(face_image.shape) == 3:
            return cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        return face_image

    def resize_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Resizes the face image to the target dimensions.

        Args:
            face_image (np.ndarray): The input face image (grayscale or BGR).

        Returns:
            np.ndarray: The resized image.
        """
        return cv2.resize(face_image, self.target_size, interpolation=cv2.INTER_AREA)

    def normalize(self, face_image: np.ndarray) -> np.ndarray:
        """
        Normalizes pixel values from [0, 255] to [0, 1].

        Args:
            face_image (np.ndarray): The input image as a numpy array.

        Returns:
            np.ndarray: The normalized image (float32).
        """
        return face_image.astype('float32') / 255.0

    def preprocess_pipeline(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Executes the full preprocessing pipeline on a face detected in a frame.

        Steps: Extract -> Grayscale -> Resize -> Normalize.

        Args:
            frame (np.ndarray): The original image/frame.
            bbox (Tuple[int, int, int, int]): The bounding box (x, y, w, h).

        Returns:
            Optional[np.ndarray]: The preprocessed face as a normalized float32 array, 
                or None if preprocessing fails.
        """
        # 1. Extract
        face = self.extract_face(frame, bbox)
        if face is None:
            return None
        
        # 2. Grayscale
        gray = self.to_grayscale(face)
        
        # 3. Resize
        resized = self.resize_face(gray)
        
        # 4. Normalize
        normalized = self.normalize(resized)
        
        return normalized

if __name__ == "__main__":
    # Internal test/demo logic
    preprocessor = FacePreprocessor(target_size=(48, 48))
    
    # Create a dummy image and box
    dummy_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    dummy_bbox = (10, 10, 50, 50)
    
    processed = preprocessor.preprocess_pipeline(dummy_img, dummy_bbox)
    
    if processed is not None:
        print(f"Preprocessed face shape: {processed.shape}")
        print(f"Data type: {processed.dtype}")
        print(f"Pixel range: [{np.min(processed)}, {np.max(processed)}]")
        print("Preprocessing module verified successfully.")
    else:
        print("Preprocessing failed for dummy inputs.")
