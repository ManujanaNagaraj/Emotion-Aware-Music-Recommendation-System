import cv2
import time
import numpy as np
from face_detection import FaceDetector
from preprocessing import FacePreprocessor
from emotion_inference import EmotionClassifier
from spotify_player import open_playlist_for_emotion
from hand_gesture_controller import HandGestureController

# Configuration for detection
CONFIDENCE_THRESHOLD = 0.4 
UI_COLORS = {
    "gesture": (255, 255, 0),    # Cyan/Yellow
    "prompt": (255, 255, 255),   # White
    "warning": (0, 0, 255),      # Red
    "success": (0, 255, 0),      # Green
    "cooldown": (50, 50, 150)    # Deep Red
}

def get_effective_emotion(detected_emotion, confidence, is_smiling=False):
    """
    Applies fallback and hybrid logic for demo purposes.
    - If the emotion is unknown but user is smiling, returns 'happy'.
    - If the emotion is unknown/low and not smiling, defaults to 'calm'.
    """
    effective = detected_emotion
    source = "AI Model"

    # Hybrid Logic: SMILE detection
    if effective == "unknown" and is_smiling:
        effective = "happy"
        source = "Hybrid (Smile Detection)"
        print(f"\n[HYBRID] Smile detected! Overriding 'unknown' with: '{effective}'")
    
    # Fallback Logic: Unknown or Low Confidence
    elif effective == "unknown" or confidence < CONFIDENCE_THRESHOLD:
        reason = "Unknown emotion" if effective == "unknown" else f"Low confidence ({confidence:.2f})"
        effective = "calm"
        source = f"Demo Fallback ({reason})"
        print(f"\n[DEMO FALLBACK] {reason} detected. Defaulting to: '{effective}'")
    
    else:
        print(f"\n[MATCH] Confidence {confidence:.2f} above threshold. Using: '{effective}'")
    
    return effective

def handle_gesture_action(gesture: str, current_emotion: str, current_confidence: float, 
                          is_smiling: bool, manual_emotion: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Processes actions based on detected gesture and returns (updated_manual_emotion, action_message)
    """
    action_message = ""
    emotions = ["happy", "sad", "angry", "calm"]
    
    if gesture == "open_palm":
        emotion_to_play = get_effective_emotion(current_emotion, current_confidence, is_smiling)
        open_playlist_for_emotion(emotion_to_play)
        action_message = f"PLAYING {emotion_to_play.upper()}"
    
    elif gesture == "point_right":
        idx = emotions.index(manual_emotion) if manual_emotion in emotions else -1
        manual_emotion = emotions[(idx + 1) % len(emotions)]
        action_message = f"SET TO {manual_emotion.upper()}"
    
    elif gesture == "point_left":
        idx = emotions.index(manual_emotion) if manual_emotion in emotions else 0
        manual_emotion = emotions[(idx - 1) % len(emotions)]
        action_message = f"SET TO {manual_emotion.upper()}"
    
    elif gesture == "fist":
        manual_emotion = None
        action_message = "RESET TO AUTO"
        
    elif gesture == "two_fingers":
        open_playlist_for_emotion(current_emotion)
        action_message = f"SHUFFLING {current_emotion.upper()} MOOD ✌️"
        
    return manual_emotion, action_message

def draw_instructions(frame: np.ndarray, gestures_enabled: bool, draw_landmarks: bool) -> None:
    """
    Draws the system control legend at the bottom of the frame.
    """
    instr_text = "Press 'p' Play | 'r' Reset | 'g' Toggle Gestures | 'l' Toggle Landmarks | 'q' Quit"
    cv2.putText(frame, instr_text, (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    # Legend for gestures (if enabled)
    if gestures_enabled:
        legend = "\u270B:Play | \ud83d\udc49:Next | \ud83d\udc48:Prev | \u270c\ufe0f:Mood | \u270A:Stop"
        cv2.putText(frame, legend, (10, frame.shape[0] - 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def run_webcam_emotion_recognition():
    """
    Captures video from the webcam, detects faces, predicts emotions, 
    and displays the results in real-time.
    """
    # 1. Initialize Components
    print("Initializing components...")
    try:
        detector = FaceDetector()
        preprocessor = FacePreprocessor(target_size=(48, 48))
        classifier = EmotionClassifier(model_path='models/emotion_cnn.h5')
        gesture_controller = HandGestureController()
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    # 2. Open Webcam
    # 0 is usually the built-in webcam. Change to 1 or 2 if you have external cameras.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return


    print("Webcam started.")
    print("--- Emotion Recognition Dashboard (Demo Mode) ---")
    print("Commands:")
    print("  'p' : Open Spotify for current emotion")
    print("  'q' : Quit application")
    print("\nManual Overrides (for testing Sad/Angry/Happy):")
    print("  'h' : Force HAPPY  |  's' : Force SAD")
    print("  'a' : Force ANGRY  |  'c' : Force CALM")
    print("  'r' : Reset to AUTO/HYBRID mode")
    print("-" * 50)

    # 3. State Tracking
    current_emotion = None
    current_confidence = 0.0
    manual_emotion = None # Manual override for demo (happy, sad, etc.)
    is_smiling = False     # Real-time smile status for hybrid detection
    gestures_enabled = True # --- NEW: Toggle for touchless controls ---
    draw_landmarks = True   # --- NEW: Toggle for hand landmark dots ---
    popup_message = ""      # --- NEW: Toast notifications ---
    popup_time = 0
    # --- SESSION STATS ---
    frames_processed = 0
    faces_detected = 0
    hands_detected = 0
    
    # 4. Real-time Loop
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip frame horizontally for a mirror effect (more natural for user)
        frame = cv2.flip(frame, 1)
        
        # Create a copy for drawing annotations
        annotated_frame = frame.copy()

        # A. Detect Faces
        frames_processed += 1
        faces = detector.detect_faces(frame)
        if len(faces) > 0:
            faces_detected += 1

        for (x, y, w, h) in faces:
            bbox = (x, y, w, h)
            
            # B. Preprocess Face
            # Returns normalized (48, 48, 1) array or None
            processed_face = preprocessor.preprocess_pipeline(frame, bbox)

            if processed_face is not None:
                # C. Predict Emotion
                label, confidence = classifier.predict(processed_face)
                
                # D. Hybrid Smile Detection (Secondary check)
                # This helps detect 'happy' even if the CNN model is missing or low-confidence
                roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                is_smiling = detector.detect_smile(roi_gray)
                
                # Hybrid Logic: If the model says 'unknown' but the user is smiling, force 'happy'
                if label == "unknown" and is_smiling:
                    label = "happy"
                    confidence = 0.95 # Simulated confidence for hybrid mode
                
                # Manual Override: If a manual emotion is set, it takes priority
                if manual_emotion:
                    label = manual_emotion
                    confidence = 1.0
                
                # Update current detected emotion
                current_emotion = label
                current_confidence = confidence
                
                # Format Label Text
                hybrid_tag = " (Hybrid)" if is_smiling and not manual_emotion else ""
                manual_tag = " (Manual)" if manual_emotion else ""
                text = f"{label}{hybrid_tag}{manual_tag}: {confidence:.2f}"
                
                # Color Coding (Optional: Change color based on emotion)
                color = (0, 255, 0) # Green for all by default
                if label == 'angry': color = (0, 0, 255) # Red
                if label == 'happy': color = (0, 255, 255) # Yellow
                
            else:
                text = "Processing Error"
                color = (0, 0, 255)

            # D. Draw Annotations
            # Rectangle around face
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Label above face
            # Ensure text doesn't go off-screen at the top
            text_y = y - 10 if y - 10 > 10 else y + h + 20
            cv2.putText(annotated_frame, text, (x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- NEW: Hand Gesture Detection ---
        gesture = "none"
        if gestures_enabled and len(faces) > 0:
            gesture, h_landmarks = gesture_controller.get_gesture(frame)
            if h_landmarks:
                hands_detected += 1
            
            if h_landmarks and draw_landmarks:
                # Draw wrist dot for tracking confirmation
                h, w, _ = frame.shape
                wrist = h_landmarks[0]
                cv2.circle(annotated_frame, (int(wrist[0]*w), int(wrist[1]*h)), 8, (0, 255, 0), -1)
                
                # Check for boundary warning
                if wrist[0] < 0.1 or wrist[0] > 0.9 or wrist[1] < 0.1 or wrist[1] > 0.9:
                    cv2.putText(annotated_frame, "MOVE HAND TO CENTER", (10, 160), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                # Draw Hand Landmarks visually on the annotated frame
                results = gesture_controller.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        gesture_controller.mp_draw.draw_landmarks(
                            annotated_frame, hand_lms, gesture_controller.mp_hands.HAND_CONNECTIONS)
            
            # Display Gesture Indicator
            if gesture != "none" and gesture != "unknown":
                # Emoji mapping for UI
                emoji_ui = {
                    "open_palm": "\u270B",
                    "fist": "\u270A",
                    "point_right": "\ud83d\udc49",
                    "point_left": "\ud83d\udc48",
                    "two_fingers": "\u270c\ufe0f"
                }
                pure_gesture = gesture.replace(" (cooldown)", "")
                icon = emoji_ui.get(pure_gesture, "")
                
                label_text = f"HAND: {pure_gesture.upper().replace('_', ' ')} {icon}"
                # Add AI Source tag
                cv2.rectangle(annotated_frame, (10, 100), (60, 115), (0, 255, 0), -1)
                cv2.putText(annotated_frame, "AI", (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                color_hand = (255, 255, 0)
                
                if "cooldown" in gesture:
                    label_text += " (LOCKED)"
                    color_hand = (50, 50, 150)
                    # Draw tiny Progress Bar for cooldown
                    cv2.rectangle(annotated_frame, (10, 135), (110, 140), (50, 50, 50), -1)
                    cv2.rectangle(annotated_frame, (10, 135), (60, 140), (0, 165, 255), -1)
                
                cv2.putText(annotated_frame, label_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hand, 2)
        elif gestures_enabled and len(faces) == 0:
            # Draw translucent red bar for suspended status
            sub_rect = annotated_frame[100:140, 0:350] # Region where HAND label or warning appears
            red_rect = np.full(sub_rect.shape, (50, 50, 150), dtype=np.uint8)
            res = cv2.addWeighted(sub_rect, 0.7, red_rect, 0.3, 0)
            annotated_frame[100:140, 0:350] = res
            
            cv2.putText(annotated_frame, "HAND GESTURES SUSPENDED (NO FACE)", (10, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # Display Gesture Toggle Status
        status_color = (0, 255, 0) if gestures_enabled else (255, 100, 100)
        status_text = "ENABLED" if gestures_enabled else "DISABLED"
        cv2.putText(annotated_frame, f"GESTURES: {status_text}", (10, annotated_frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        if not gestures_enabled:
            # Draw banner for Keyboard Only mode
            cv2.rectangle(annotated_frame, (0, annotated_frame.shape[0]-95), (150, annotated_frame.shape[0]-75), (255, 0, 0), -1)
            cv2.putText(annotated_frame, "KEYBOARD ONLY", (10, annotated_frame.shape[0]-80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        l_color = (0, 255, 255) if draw_landmarks else (100, 100, 100)
        l_text = "SHOWING" if draw_landmarks else "HIDDEN ('l' to toggle)"
        cv2.putText(annotated_frame, f"LANDMARKS: {l_text}", (10, annotated_frame.shape[0] - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, l_color, 1)

        # --- Global Overrides & Indicators ---
        if manual_emotion:
            cv2.putText(annotated_frame, "DEMO MODE: MANUAL OVERRIDE ACTIVE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Locked to: {manual_emotion.upper()}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Display Instructions on frame
        draw_instructions(annotated_frame, gestures_enabled, draw_landmarks)

        # Display Result
        # --- NEW: Action Popup Notification ---
        if popup_message and time.time() - popup_time < 2.0:
            # Draw semi-transparent background for popup
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], 50), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
            cv2.putText(annotated_frame, popup_message, (annotated_frame.shape[1]//2 - 100, 35), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Emotion Recognition', annotated_frame)

        # Exit Strategy
        key = cv2.waitKey(1) & 0xFF
        
        # --- NEW: Map Gestures to Actions ---
        if gesture != "none" and "cooldown" not in gesture:
            manual_emotion, msg = handle_gesture_action(
                gesture, current_emotion, current_confidence, is_smiling, manual_emotion
            )
            if msg:
                popup_message = msg
                popup_time = time.time()

        if key == ord('q'):
            break
        elif key == ord('g'):
            gestures_enabled = not gestures_enabled
            print(f"\n[CONTROL] Hand Gestures: {'ENABLED' if gestures_enabled else 'DISABLED'}")
        elif key == ord('l'):
            draw_landmarks = not draw_landmarks
            print(f"\n[CONTROL] Hand Landmarks: {'SHOWN' if draw_landmarks else 'HIDDEN'}")
        elif key == ord('h'):
            manual_emotion = "happy"
            print("\n[DEMO OVERRIDE] Manual emotion set to: HAPPY")
        elif key == ord('s'):
            manual_emotion = "sad"
            print("\n[DEMO OVERRIDE] Manual emotion set to: SAD")
        elif key == ord('a'):
            manual_emotion = "angry"
            print("\n[DEMO OVERRIDE] Manual emotion set to: ANGRY")
        elif key == ord('c'):
            manual_emotion = "calm"
            print("\n[DEMO OVERRIDE] Manual emotion set to: CALM")
        elif key == ord('r'):
            manual_emotion = None
            print("\n[DEMO RESET] Overrides cleared. System in AUTO/HYBRID mode.")
        elif key == ord('p'):
            if current_emotion:
                # Determine emotion with fallback logic
                emotion_to_play = get_effective_emotion(current_emotion, current_confidence, is_smiling)
                
                print(f"[ACTION] Launching Spotify (App/Web) for: {emotion_to_play}")
                open_playlist_for_emotion(emotion_to_play)
            else:
                print("\n[WARNING] 'p' pressed but no emotion detected yet.")

    # Cleanup
    print("\n" + "="*40)
    print("SESSION SUMMARY")
    print("="*40)
    print(f"Total Frames Processed: {frames_processed}")
    print(f"Frames with Faces:      {faces_detected} ({faces_detected/max(1,frames_processed)*100:.1f}%)")
    print(f"Frames with Hands:      {hands_detected} ({hands_detected/max(1,frames_processed)*100:.1f}%)")
    print("="*40 + "\n")
    
    cap.release()
    cv2.destroyAllWindows()
    gesture_controller.release()
    print("Resources released. Exiting application.")

if __name__ == "__main__":
    run_webcam_emotion_recognition()
