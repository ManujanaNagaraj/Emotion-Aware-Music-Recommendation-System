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

    def _is_finger_open(self, landmarks: List[Tuple], finger_tip_idx: int) -> bool:
        """
        Determines if a finger is open by comparing tip height with PIP joint height.
        Tip landmarks: Index=8, Middle=12, Ring=16, Pinky=20
        """
        # For non-thumb fingers, tip Y should be smaller (higher on screen) than PIP joint Y
        # PIP joints are at tip_idx - 2 (Approximate MediaPipe indexing)
        return landmarks[finger_tip_idx][1] < landmarks[finger_tip_idx - 2][1]

    def _is_thumb_open(self, landmarks: List[Tuple]) -> bool:
        """
        Special detection for thumb which typically moves horizontally.
        Tip=4, IP Joint=3, MCP Joint=2
        """
        # For a right hand facing camera, thumb tip X should be smaller than MCP X 
        # (This varies by hand orientation, but we'll use a simple proximity check for now)
        return abs(landmarks[4][0] - landmarks[17][0]) > abs(landmarks[3][0] - landmarks[17][0])

    def get_finger_states(self, landmarks: List[Tuple]) -> List[bool]:
        """
        Returns a list of 5 booleans [Thumb, Index, Middle, Ring, Pinky]
        True = Open, False = Closed
        """
        return [
            self._is_thumb_open(landmarks),
            self._is_finger_open(landmarks, 8),   # Index
            self._is_finger_open(landmarks, 12),  # Middle
            self._is_finger_open(landmarks, 16),  # Ring
            self._is_finger_open(landmarks, 20)   # Pinky
        ]

    def classify_gesture(self, finger_states: List[bool]) -> str:
        """
        Maps a list of finger booleans to a gesture string.
        """
        open_count = sum(finger_states)
        
        # 1. Open Palm: All 5 fingers up
        if open_count == 5:
            return "open_palm"
            
        # 2. Fist: All 5 fingers down
        if open_count == 0:
            return "fist"
            
        # 3. Two Fingers (Victory/Next): Index and Middle up
        if open_count == 2 and finger_states[1] and finger_states[2]:
            return "two_fingers"
            
        # 4. Thumb Up: Only thumb is up
        if open_count == 1 and finger_states[0]:
            return "thumb_up"
            
        return "unknown"

    def get_gesture(self, frame) -> Tuple[str, Optional[List]]:
        """
        Orchestrates detection and classification.
        Returns (gesture_name, landmarks)
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_landmarks(frame_rgb)
        
        if landmarks:
            finger_states = self.get_finger_states(landmarks)
            gesture = self.classify_gesture(finger_states)
            return gesture, landmarks
        
        return "none", None

if __name__ == "__main__":
    # Standalone Test Mode
    controller = HandGestureController()
    cap = cv2.VideoCapture(0)
    
    print("--- Hand Gesture Controller Test ---")
    print("Commands: 'q' to quit")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        gesture, landmarks = controller.get_gesture(frame)
        
        if landmarks:
            # Optional: Simple terminal output for verification
            if gesture != "unknown":
                print(f"Detected Gesture: {gesture.upper()}")
                
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Gesture Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
