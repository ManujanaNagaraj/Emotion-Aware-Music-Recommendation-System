"""
Centralized Emotion Configuration

This module serves as the single source of truth for all emotion-related labels
and their corresponding numeric mappings. Centralizing this configuration 
prevents inconsistencies across the data loading, model training, and 
real-time inference stages of the pipeline.
"""

from typing import List, Dict

# List of supported emotion labels in the system
EMOTIONS: List[str] = [
    "happy",
    "sad",
    "calm",
    "angry"
]

# Mapping from emotion label to numeric class index
# This is used for labeling training data and model output interpretation
EMOTION_TO_INDEX: Dict[str, int] = {
    emotion: index for index, emotion in enumerate(EMOTIONS)
}

# Reverse mapping from numeric index back to emotion label
# This is used during real-time inference to convert model predictions to human-readable text
INDEX_TO_EMOTION: Dict[int, str] = {
    index: emotion for emotion, index in EMOTION_TO_INDEX.items()
}

# Target image size for the emotion classification model (standard for FER-2013)
TARGET_IMAGE_SIZE = (48, 48)

if __name__ == "__main__":
    # Internal validation/demo logic
    print("--- Emotion Configuration Summary ---")
    print(f"Supported Emotions: {EMOTIONS}")
    print(f"Emotion to Index: {EMOTION_TO_INDEX}")
    print(f"Index to Emotion: {INDEX_TO_EMOTION}")
    print(f"Target Image Size: {TARGET_IMAGE_SIZE}")
    
    # Simple consistency check
    assert len(EMOTION_TO_INDEX) == len(INDEX_TO_EMOTION), "Mapping size mismatch!"
    print("\nConfiguration module verified: Mappings are consistent.")
