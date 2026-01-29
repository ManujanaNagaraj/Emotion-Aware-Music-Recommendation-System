import mediapipe as mp
try:
    hands = mp.solutions.hands.Hands()
    print("MediaPipe Hands initialized successfully!")
    hands.close()
except Exception as e:
    print(f"MediaPipe Init Failed: {e}")
