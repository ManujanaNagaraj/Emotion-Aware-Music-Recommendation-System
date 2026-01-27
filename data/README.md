# Facial Emotion Recognition Dataset

This directory structure is designed to house image data for training and evaluating the Emotion-Aware Music Recommendation System.

## Dataset Purpose
The primary goal of this dataset is to provide a balanced collection of facial expressions categorized by emotional state. These categories (Happy, Sad, Calm, Angry) serve as the foundation for the emotion detection engine, which in turn drives musical recommendations.

## Folder Organization
The dataset is split into three main subsets:
- `train/`: Used for model training.
- `val/`: Used for hyperparameter tuning and early stopping.
- `test/`: Used for final evaluation of model performance.

Inside each split, images are organized into subdirectories named after their respective emotion classes:
- `happy/`
- `sad/`
- `calm/`
- `angry/`

## Guidelines for Data Organization
1. **Format**: Preferred formats are `.jpg`, `.jpeg`, or `.png`.
2. **Resolution**: Images should ideally be cropped to focus on the face. Square resolutions (e.g., 48x48, 128x128, or 224x224) are recommended depending on the model architecture.
3. **Consistency**: Ensure that images of the same person across different emotions are distributed fairly across splits to avoid data leakage (if applicable).
4. **Balancing**: Aim for a similar number of images per class to prevent model bias.

## How to Add Data
To add new data, simply place your images into the corresponding emotion folder within the appropriate split. For example:
`data/train/happy/user_123.jpg`
