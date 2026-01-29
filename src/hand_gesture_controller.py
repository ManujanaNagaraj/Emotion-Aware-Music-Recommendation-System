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
    
    Supported Gestures:
    - Open Palm (✋): Play current emotion playlist
    - Fist (✊): Stop/Reset to Auto mode
    - Point Right (👉): Next mood/track
    - Point Left (👈): Previous mood/track
    - Two Fingers (✌️): Shuffle/Re-open current mood
    
    MediaPipe Landmark Indexing Reference:
    0: Wrist
    1-4: Thumb (MCP, IP, TIP)
    5-8: Index (MCP, PIP, DIP, TIP)
    9-12: Middle (MCP, PIP, DIP, TIP)
    13-16: Ring (MCP, PIP, DIP, TIP)
    17-20: Pinky (MCP, PIP, DIP, TIP)
    """
    def __init__(self):
        # --- NEW: MediaPipe Tasks API for Python 3.13 Support ---
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            model_path = 'models/hand_landmarker.task'
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_tasks_api = True
            print("[INFO] HandGestureController initialized using MediaPipe Tasks API.")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize MediaPipe Tasks API: {e}")
            self.detector = None
            self.use_tasks_api = False
        
        # Gesture Lifecycle State
        self.last_gesture = "none"
        self.last_trigger_time = 0
        self.cooldown_seconds = 2.0
        self.terminal_debug = True  # Toggle for console logging

    def release(self):
        """
        Properly releases MediaPipe resources.
        """
        if self.use_tasks_api and self.detector:
            self.detector.close()

    def _extract_landmarks(self, frame_rgb) -> Optional[List[Tuple[float, float, float]]]:
        """
        Processes frame and returns a list of landmark coordinates (x, y, z).
        """
        if not self.use_tasks_api or self.detector is None:
            return None
            
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            # We only support one hand for this demo
            hand_landmarks = detection_result.hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
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
        Special detection for thumb.
        Tip=4, IP Joint=3, Index Base=5, Pinky Base=17
        """
        # Distance between thumb tip and index base (Landmark 5)
        # If the thumb is folded across the palm, it gets close to landmark 5
        distance_tip = ((landmarks[4][0]-landmarks[5][0])**2 + (landmarks[4][1]-landmarks[5][1])**2)**0.5
        distance_ip = ((landmarks[3][0]-landmarks[5][0])**2 + (landmarks[3][1]-landmarks[5][1])**2)**0.5
        
        return distance_tip > distance_ip

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

    def classify_gesture(self, finger_states: List[bool], landmarks: List[Tuple]) -> str:
        """
        Maps a list of finger booleans and landmarks to a gesture string.
        """
        open_count = sum(finger_states)
        thumb_open, index_open, middle_open, ring_open, pinky_open = finger_states
        
        # 1. Open Palm (✋): 4 or 5 fingers up
        if open_count >= 4:
            return "open_palm"
            
        # 2. Fist (✊): All fingers down
        if open_count == 0:
            return "fist"
            
        # 3. Two Fingers (✌️): Index and Middle up
        if open_count == 2 and index_open and middle_open:
            return "two_fingers"

        # 4. Pointing Logic (👉/👈): Only Index finger up
        if open_count == 1 and index_open:
            index_tip_x = landmarks[8][0]
            index_mcp_x = landmarks[5][0]
            
            # Point Right: Tip is further right than MCP
            if index_tip_x > index_mcp_x + 0.05: # threshold for intent
                return "point_right"
            # Point Left: Tip is further left than MCP
            elif index_tip_x < index_mcp_x - 0.05:
                return "point_left"
            
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
            gesture = self.classify_gesture(finger_states, landmarks)
            
            # Diagnostic for terminal debug
            if self.terminal_debug:
                f_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                active = [f_names[i] for i, s in enumerate(finger_states) if s]
                print(f"[DEBUG] Open Fingers: {active} | Count: {sum(finger_states)}")
            
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
            # Draw dots on fingers classified as OPEN
            finger_states = controller.get_finger_states(landmarks)
            tip_ids = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
            h, w, _ = frame.shape
            for i, is_open in enumerate(finger_states):
                if is_open:
                    lm = landmarks[tip_ids[i]]
                    cv2.circle(frame, (int(lm[0]*w), int(lm[1]*h)), 10, (0, 0, 255), -1)

            # Draw landmarks in test mode using MediaPipe utilities
            results = controller.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    controller.mp_draw.draw_landmarks(
                        frame, hand_lms, controller.mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, "NO HAND DETECTED", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        # Map gesture names to emojis for display
        emoji_map = {
            "open_palm": "OPEN PALM (PLAY) \u270B",
            "fist": "FIST (STOP) \u270A",
            "point_right": "POINT RIGHT (NEXT) \ud83d\udc49",
            "point_left": "POINT LEFT (PREV) \ud83d\udc48",
            "two_fingers": "TWO FINGERS (MOOD) \u270c\ufe0f",
            "none": "NONE",
            "unknown": "UNKNOWN"
        }
        display_gesture = emoji_map.get(gesture.split(" (")[0], gesture)

        cv2.putText(frame, f"Gesture: {display_gesture}", (10, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display small instructions
        cv2.putText(frame, "Touchless Demo Mode | Press 'q' to exit", (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Gesture Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
