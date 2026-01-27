import cv2
from face_detection import FaceDetector

def run_webcam_face_detection():
    """
    Captures live video from the webcam and performs real-time face detection
    using the FaceDetector module.
    
    Press 'q' to exit the video feed.
    """
    # Initialize the face detector
    try:
        detector = FaceDetector()
        print("Success: FaceDetector initialized.")
    except Exception as e:
        print(f"Error: Failed to initialize face detector. {e}")
        return

    # Open a connection to the primary webcam (index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time face detection. Press 'q' to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from webcam.")
            break

        # Detect faces in the current frame
        faces = detector.detect_faces(frame)

        # Draw bounding boxes around the faces
        for (x, y, w, h) in faces:
            # Draw rectangle: (image, top-left, bottom-right, color, thickness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Face Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam feed stopped.")

if __name__ == "__main__":
    run_webcam_face_detection()
