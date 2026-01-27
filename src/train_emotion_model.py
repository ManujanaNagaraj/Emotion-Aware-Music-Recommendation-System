import os
import tensorflow as tf
from emotion_model import build_emotion_model
from emotion_config import EMOTIONS, TARGET_IMAGE_SIZE

def compile_model(model, learning_rate=0.001):
    """
    Compiles the Keras model with standard optimization and loss functions
    for multi-class classification.

    Args:
        model (tf.keras.Model): The emotion detection model instance.
        learning_rate (float): The step size for the Adam optimizer.

    Returns:
        tf.keras.Model: The compiled model.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def get_train_callbacks(checkpoint_path: str):
    """
    Defines standard Keras callbacks for training efficiency and model persistence.

    Args:
        checkpoint_path (str): Directory where the best model weights will be saved.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of configured callbacks.
    """
    callbacks = [
        # Save the best version of the model based on validation loss
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, 'best_emotion_model.h5'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        # Stop training early if validation loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when a plateau is reached
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    return callbacks

def train_model():
    """
    Main orchestration function to set up hyperparameters and initiate 
    the training loop.
    """
    # --- Training Configuration ---
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = len(EMOTIONS)
    INPUT_SHAPE = (TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 1)
    CHECKPOINT_DIR = 'models/checkpoints'

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # 1. Build the model architecture
    model = build_emotion_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    
    # 2. Compile with optimizer and loss
    model = compile_model(model, learning_rate=LEARNING_RATE)

    print("\n--- Model Compilation Complete ---")
    model.summary()

    # 3. Data Loading (Placeholders)
    # TODO: Integrate EmotionDataLoader to fetch training and validation generators
    # train_generator = ...
    # val_generator = ...
    
    print("\n[TODO]: Data loaders are not yet connected. Connect training generators here.")

    # 4. Initiate Training
    # TODO: Uncomment and run once data loaders are ready
    """
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=val_generator.samples // BATCH_SIZE,
        callbacks=get_train_callbacks(CHECKPOINT_DIR)
    )
    print("\nTraining completed.")
    """

if __name__ == "__main__":
    # Guarded execution to define the pipeline without running it by default
    print("Emotion Awareness Music Recommendation System - Training Module")
    print("---------------------------------------------------------------")
    
    # Note: Calling train_model() here will initialize the architecture and configuration
    # but actual fit() is commented out awaiting real data integration.
    train_model()
