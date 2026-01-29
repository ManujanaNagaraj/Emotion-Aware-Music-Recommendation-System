import sys
import os

def verify_dependencies():
    print("--- Environment Verification ---")
    
    dependencies = [
        "cv2",
        "numpy",
        "tensorflow",
        "h5py"
    ]
    
    all_found = True
    for lib in dependencies:
        try:
            __import__(lib)
            print(f"[SUCCESS] {lib} is installed.")
        except ImportError:
            print(f"[FAILURE] {lib} is MISSING.")
            all_found = False
            
    if not all_found:
        print("\n[ERROR] Some dependencies are missing. Run: pip install -r requirements.txt")
        return False

    print("\n--- Model Verification ---")
    model_path = 'models/emotion_cnn.h5'
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            print(f"[SUCCESS] Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"[FAILURE] Error loading model: {e}")
            return False
    else:
        print(f"[WARNING] Model file not found at {model_path}. This is expected if the model hasn't been trained yet.")

    print("\n[COMPLETE] Environment is ready.")
    return True

if __name__ == "__main__":
    success = verify_dependencies()
    sys.exit(0 if success else 1)
