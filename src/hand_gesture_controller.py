"""
Hand Gesture Controller Module
Enables touchless control of the music system using MediaPipe Hands.

Example:
    controller = HandGestureController()
    gesture, landmarks = controller.get_gesture(frame)
    if gesture == "open_palm":
        # Trigger action
"""
import cv2
import mediapipe as mp
import time
from typing import List, Tuple, Optional

class HandGestureController:
    """
    Handles hand landmark detection and gesture classification using MediaPipe.
    
    MediaPipe Landmark Indexing Reference:
    0: Wrist
    1-4: Thumb (MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
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
        
        # Gesture Lifecycle State
        self.last_gesture = "none"
        self.last_trigger_time = 0
        self.cooldown_seconds = 2.0

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
        Tip=4, IP Joint=3, MCP Joint=2, Wrist=0
        
        Note: This assumes the hand is facing the camera (palm forward).
        """
        # Distance between thumb tip and Pinky MCP (Landmark 17)
        # If open, the thumb tip should be further from the hand base than the IP joint
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
        if frame is None or frame.size == 0:
            return "none", None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self._extract_landmarks(frame_rgb)
        
        if landmarks:
            finger_states = self.get_finger_states(landmarks)
            gesture = self.classify_gesture(finger_states)
            
            # Cooldown logic: Only return a gesture if enough time has passed
            current_time = time.time()
            if gesture != "none" and gesture != "unknown":
                if current_time - self.last_trigger_time > self.cooldown_seconds:
                    self.last_trigger_time = current_time
                    self.last_gesture = gesture
                    return gesture, landmarks
                else:
                    return f"{gesture} (cooldown)", landmarks
            
            return gesture, landmarks
        
        return "none", None

if __name__ == "__main__":
    # Standalone Test Mode
    controller = HandGestureController()
    cap = cv2.VideoCapture(0)
    
    print("--- Hand Gesture Controller Test ---")
    print("Commands: 'q' to quit")
    
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        gesture, landmarks = controller.get_gesture(frame)
        
        if landmarks:
            # Draw landmarks in test mode using MediaPipe utilities
            results = controller.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    controller.mp_draw.draw_landmarks(
                        frame, hand_lms, controller.mp_hands.HAND_CONNECTIONS)
                
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.imshow('Gesture Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
