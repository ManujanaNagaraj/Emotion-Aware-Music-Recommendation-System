import os
import numpy as np
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
from typing import Tuple, Optional, Union

# Import configuration
# Ensure this module is run from a context where 'src' is in pythonpath or relative import works
try:
    from emotion_config import INDEX_TO_EMOTION, TARGET_IMAGE_SIZE
except ImportError:
    # Fallback for direct script execution if needed, though usually run as module
    from src.emotion_config import INDEX_TO_EMOTION, TARGET_IMAGE_SIZE

class EmotionClassifier:
    """
    A unified interface for loading the trained Keras model and performing 
    emotion classification on preprocessed face images.
    """

    def __init__(self, model_path: str = 'models/emotion_cnn.h5'):
        """
        Initializes the classifier by loading the trained model.

        Args:
            model_path (str): Path to the .h5 model file.
        """
        self.model = None
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            print(f"Bypass Warning: Model file not found at {model_path}. Inference will return dummies.")
            return

        if not TENSORFLOW_AVAILABLE:
            print("[CRITICAL WARNING] TensorFlow is not installed. Inference will return 'unknown'.")
            print("Note: TensorFlow does not yet support Python 3.13 on Windows.")
            return

        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Successfully loaded emotion model from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def predict(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Predicts the emotion of a given face image.

        Args:
            face_image (np.ndarray): A preprocessed face image. 
                                     Expected shape: (48, 48) or (48, 48, 1).
                                     Values should be normalized to [0, 1].

        Returns:
            Tuple[str, float]: The predicted emotion label and its confidence score.
                               Returns ('unknown', 0.0) if model is not loaded or input is invalid.
        """
        if self.model is None:
            return "unknown", 0.0
            
        if face_image is None or face_image.size == 0:
            return "unknown", 0.0

        # Ensure input shape is (1, 48, 48, 1)
        # 1. Add channel dimension if missing (48, 48) -> (48, 48, 1)
        if len(face_image.shape) == 2:
            face_image = np.expand_dims(face_image, axis=-1)
            
        # 2. Add batch dimension (48, 48, 1) -> (1, 48, 48, 1)
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)

        try:
            # Perform inference
            predictions = self.model.predict(face_image, verbose=0)
            
            # Extract result
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            
            emotion_label = INDEX_TO_EMOTION.get(predicted_index, "unknown")
            
            return emotion_label, confidence

        except Exception as e:
            print(f"Inference error: {e}")
            return "error", 0.0

if __name__ == "__main__":
    # Test Block
    print("--- Testing EmotionClassifier ---")
    
    # Path assumption: running from repo root
    classifier = EmotionClassifier(model_path='models/emotion_cnn.h5')
    
    # Create a dummy input matching the expected input shape (48, 48, 1)
    # Normalized to ranges [0,1]
    dummy_face = np.random.rand(48, 48, 1).astype(np.float32)
    
    emotion, conf = classifier.predict(dummy_face)
    
    print(f"Dummy Prediction Result: Emotion='{emotion}', Confidence={conf:.4f}")
