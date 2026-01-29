# Emotion-Aware Music Recommendation System

An intelligent, real-time music recommendation system that leverages Computer Vision and Machine Learning to suggest Spotify playlists based on the user's facial expressions.

## 📌 Project Overview
The goal of this system is to bridge the gap between affective computing and digital entertainment. By capturing human emotions through a webcam, the system translates facial expressions into personalized musical recommendations, providing an empathetic listening experience.

## 🔄 System Workflow
The project follows a modular pipeline to deliver real-time recommendations:
1.  **Webcam Capture**: Accesses the local camera to stream live video frames.
2.  **Face Detection**: Identifies faces within the frame using a Haar Cascade/MTCNN-based detector.
3.  **Emotion Classification**: Processes the cropped face using a pre-trained CNN model to predict the dominant emotion (e.g., Happy, Sad, Calm, Angry).
4.  **Playlist Mapping**: Maps the detected emotion label to a curated Spotify playlist ID.
5.  **Spotify Launch**: Automatically constructs a Spotify web URL and opens it in the default browser upon user trigger.

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow / Keras (CNN Architecture)
- **Signal Processing**: NumPy, Pandas (for data handling)
- **Integration**: Python `webbrowser` module for **Dual Spotify Redirection** (Native App + Web).

## ✨ Key Features
- **Real-time Emotion Recognition**: Continuous monitoring and classification of facial expressions.
- **Emotion-Aware Mapping**: Specialized playlists curated for specific emotional states.
- **User-Controlled Redirection**: Trigger playlists only when you want them. The system intelligently switches between the Desktop App and Browser for maximum reliability.
- **Modular Design**: Clean separation between inference, configuration, and playback logic.

## 🚀 How to Run the Project

### 1. Environment Setup
Create and activate a virtual environment to manage dependencies:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 2. Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### 3. Start Webcam Emotion Detection
Run the main pipeline script:
```bash
python src/webcam_emotion_detection.py
```

## 🎮 Demo Instructions
Once the webcam window is active, use the following keyboard controls:
- **`p`**: Press to open the Spotify playlist mapped to the **current detected emotion**.
- **`q`**: Press to quit the application and close all windows.

---
*Note: This project is designed for professional demonstration of AI and Computer Vision capabilities. It uses direct web redirection and does not require Spotify API credentials.*
