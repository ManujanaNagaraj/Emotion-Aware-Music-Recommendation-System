import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from preprocessing import FacePreprocessor
from emotion_config import EMOTIONS, EMOTION_TO_INDEX, TARGET_IMAGE_SIZE

class EmotionDataLoader:
    """
    A class for loading and managing the Emotion Detection dataset.
    
    It scans the directory structure, loads images, applies preprocessing
    (grayscale, resize, normalize), and returns numpy arrays ready for training.
    """

    def __init__(self, data_dir: str):
        """
        Initializes the data loader with the root dataset directory.

        Args:
            data_dir (str): Path to the root 'data/' directory.
        """
        self.data_dir = data_dir
        self.emotions = EMOTIONS
        self.preprocessor = FacePreprocessor(target_size=TARGET_IMAGE_SIZE)

    def load_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Loads images and labels for a specific data split.

        Args:
            split (str): One of 'train', 'val', or 'test'.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - X: Image data array of shape (N, 48, 48, 1)
                - y: Label data array of shape (N,)
        """
        images = []
        labels = []
        
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory not found: {split_dir}")
            return np.array([]), np.array([])

        print(f"Loading {split} data from {split_dir}...")

        for emotion in self.emotions:
            emotion_dir = os.path.join(split_dir, emotion)
            
            # Skip if specific emotion folder doesn't exist
            if not os.path.exists(emotion_dir):
                continue
            
            label_index = EMOTION_TO_INDEX[emotion]
            file_list = os.listdir(emotion_dir)
            
            for img_name in file_list:
                img_path = os.path.join(emotion_dir, img_name)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    # Preprocess Pipeline
                    # 1. Convert to grayscale (if not already)
                    gray = self.preprocessor.to_grayscale(img)
                    
                    # 2. Resize to target dimension (e.g., 48x48)
                    resized = self.preprocessor.resize_face(gray)
                    
                    # 3. Normalize pixel values to [0, 1]
                    normalized = self.preprocessor.normalize(resized)
                    
                    # 4. Expand dimensions to (48, 48, 1) for Keras input
                    if len(normalized.shape) == 2:
                        normalized = np.expand_dims(normalized, axis=-1)
                        
                    images.append(normalized)
                    labels.append(label_index)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} images for split '{split}'.")
        return X, y

if __name__ == "__main__":
    # Internal test
    loader = EmotionDataLoader(data_dir="./data")
    print(f"Initialized EmotionDataLoader for: {loader.emotions}")
    
    # Attempt to load a split (will be empty if no data exists yet)
    X, y = loader.load_data("train")
    if len(X) > 0:
        print(f"Sample shape: {X.shape}, Label shape: {y.shape}")
