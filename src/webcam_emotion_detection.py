import cv2
import time
import numpy as np
from face_detection import FaceDetector
from preprocessing import FacePreprocessor
from emotion_inference import EmotionClassifier
from spotify_player import open_playlist_for_emotion
from hand_gesture_controller import HandGestureController

# Configuration for detection
CONFIDENCE_THRESHOLD = 0.4  # Threshold below which emotion defaults to 'calm'

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
        # Returns list of (x, y, w, h)
        faces = detector.detect_faces(frame)

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
        if gestures_enabled:
            gesture, h_landmarks = gesture_controller.get_gesture(frame)
            
            if h_landmarks:
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
                if "cooldown" in gesture:
                    label_text += " (WAIT)"
                
                cv2.putText(annotated_frame, label_text, (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display Gesture Toggle Status
        status_color = (0, 255, 0) if gestures_enabled else (100, 100, 100)
        status_text = "ENABLED" if gestures_enabled else "DISABLED ('g' to toggle)"
        cv2.putText(annotated_frame, f"GESTURES: {status_text}", (10, annotated_frame.shape[0] - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # --- Global Overrides & Indicators ---
        if manual_emotion:
            cv2.putText(annotated_frame, "DEMO MODE: MANUAL OVERRIDE ACTIVE", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Locked to: {manual_emotion.upper()}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Display Instructions on frame
        instr_text = "Press 'p' Play | 'r' Reset | 'g' Toggle Gestures | 'q' Quit"
        cv2.putText(annotated_frame, instr_text, (10, annotated_frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Display Result
        cv2.imshow('Emotion Recognition', annotated_frame)

        # Exit Strategy
        key = cv2.waitKey(1) & 0xFF
        
        # --- NEW: Map Gestures to Actions ---
        if gesture == "open_palm":
            emotion_to_play = get_effective_emotion(current_emotion, current_confidence, is_smiling)
            print(f"\n[GESTURE] OPEN_PALM -> Triggering Spotify: {emotion_to_play}")
            open_playlist_for_emotion(emotion_to_play)
        
        elif gesture == "point_right" or gesture == "point_right (cooldown)":
            if "cooldown" not in gesture:
                emotions = ["happy", "sad", "angry", "calm"]
                idx = emotions.index(manual_emotion) if manual_emotion in emotions else -1
                manual_emotion = emotions[(idx + 1) % len(emotions)]
                print(f"\n[GESTURE] POINT_RIGHT -> Next Emotion: {manual_emotion}")
        
        elif gesture == "point_left" or gesture == "point_left (cooldown)":
            if "cooldown" not in gesture:
                emotions = ["happy", "sad", "angry", "calm"]
                idx = emotions.index(manual_emotion) if manual_emotion in emotions else 0
                manual_emotion = emotions[(idx - 1) % len(emotions)]
                print(f"\n[GESTURE] POINT_LEFT -> Prev Emotion: {manual_emotion}")
        
        elif gesture == "fist":
            manual_emotion = None
            print("\n[GESTURE] FIST -> Resetting to Auto Mode")
        
        elif gesture == "two_fingers":
            print(f"\n[GESTURE] TWO_FINGERS -> Shuffling {current_emotion} playlist")
            open_playlist_for_emotion(current_emotion)

        if key == ord('q'):
            break
        elif key == ord('g'):
            gestures_enabled = not gestures_enabled
            print(f"\n[CONTROL] Hand Gestures: {'ENABLED' if gestures_enabled else 'DISABLED'}")
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
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released. Exiting application.")

if __name__ == "__main__":
    run_webcam_emotion_recognition()
