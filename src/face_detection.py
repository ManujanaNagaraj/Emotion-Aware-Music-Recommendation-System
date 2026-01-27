import cv2
import os
from typing import List, Tuple

class FaceDetector:
    """
    A class to handle face detection using OpenCV's Haar Cascade classifier.
    
    This module provides a lightweight way to identify face bounding boxes
    in image frames, serving as the first stage of the emotion detection pipeline.
    """

    def __init__(self, model_path: str = None):
        """
        Initializes the FaceDetector by loading the Haar Cascade model.

        Args:
            model_path (str, optional): Path to the Haar Cascade XML file. 
                If None, it defaults to the OpenCV built-in frontal face model.
        
        Raises:
            FileNotFoundError: If the cascade file cannot be found or loaded.
        """
        if model_path is None:
            # Default to OpenCV's built-in frontal face cascade
            model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Haar Cascade model not found at: {model_path}")

        self.face_cascade = cv2.CascadeClassifier(model_path)

        if self.face_cascade.empty():
            raise FileNotFoundError(f"Failed to load Haar Cascade model from: {model_path}")

    def detect_faces(self, frame) -> List[Tuple[int, int, int, int]]:
        """
        Detects faces in the provided image frame.

        Args:
            frame (numpy.ndarray): The input image/frame in BGR or Grayscale format.

        Returns:
            List[Tuple[int, int, int, int]]: A list of bounding boxes, 
                where each box is (x, y, width, height).
        """
        if frame is None:
            return []

        # Convert to grayscale for Haar Cascade processing
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame

        # Detect faces
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        # minSize: Minimum possible object size. Objects smaller than that are ignored.
        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert result to list of tuples for consistency
        return [tuple(f) for f in faces]

if __name__ == "__main__":
    # Internal test/demo logic
    try:
        detector = FaceDetector()
        print("FaceDetector initialized successfully with default Haar Cascade.")
        
        # Create a dummy black image to test detection logic (should return empty list)
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detected = detector.detect_faces(dummy_frame)
        print(f"Test Detection on empty frame: {detected}")
        
    except Exception as e:
        print(f"Error during FaceDetector initialization: {e}")
