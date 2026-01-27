import tensorflow as tf
from tensorflow.keras import layers, models

def build_emotion_model(input_shape=(48, 48, 1), num_classes=4):
    """
    Builds a baseline Convolutional Neural Network (CNN) for facial emotion classification.
    
    The architecture is designed to extract hierarchical spatial features from 
    grayscale face images and classify them into one of the target emotion categories.
    
    Args:
        input_shape (tuple): The shape of the input image (height, width, channels).
            Standard for FER datasets is (48, 48, 1).
        num_classes (int): Number of target emotion classes (e.g., Happy, Sad, Calm, Angry).
        
    Returns:
        tf.keras.Model: A compiled-ready Keras model instance.
    """
    model = models.Sequential(name="Emotion_Detection_CNN")

    # --- Block 1: Initial Feature Extraction ---
    # 32 filters, 3x3 kernel. Extract basic edges and textures.
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Block 2: Deeper Features ---
    # 64 filters. Extract more complex facial features.
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Block 3: High-Level Concepts ---
    # 128 filters. Capture subtle spatial relationships.
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    # --- Classifier ---
    model.add(layers.Flatten())
    
    # Dense layer for feature integration
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    
    # Output layer with Softmax for probability distribution across classes
    model.add(layers.Dense(num_classes, activation='softmax', name="output_layer"))

    return model

if __name__ == "__main__":
    # Internal test: Build and summarize the model
    try:
        # Import emotion config to align parameters if needed in future
        # from emotion_config import EMOTIONS, TARGET_IMAGE_SIZE
        
        # Build model
        cnn_model = build_emotion_model(input_shape=(48, 48, 1), num_classes=4)
        
        # Display architecture
        print("\n--- Emotion Classification CNN Architecture ---")
        cnn_model.summary()
        print("\nModel built successfully and architecture verified.")
        
    except Exception as e:
        print(f"Error during model building: {e}")
