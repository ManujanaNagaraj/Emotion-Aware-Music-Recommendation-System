import cv2
import mediapipe as mp
import time
from typing import List, Tuple, Optional

class HandGestureController:
    """
    Handles hand landmark detection and gesture classification using MediaPipe.
    """
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
