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

    def _extract_landmarks(self, frame_rgb) -> Optional[List[Tuple[float, float, float]]]:
        """
        Processes frame and returns a list of landmark coordinates (x, y, z).
        """
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            # We only support one hand for this demo
            hand_landmarks = results.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return None
