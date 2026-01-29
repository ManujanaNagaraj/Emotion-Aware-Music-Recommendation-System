"""
Spotify Player Module

This module handles the logic for opening Spotify playlists in a web browser
based on detected emotions, using the configuration defined in `spotify_config.py`.
"""

import webbrowser
import logging
from spotify_config import EMOTION_PLAYLISTS, get_playlist_url, get_playlist_uri

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def open_playlist_for_emotion(emotion: str) -> None:
    """
    Validates the emotion, converts the corresponding playlist ID to a URL,
    and opens it in the default web browser.

    Args:
        emotion (str): The detected emotion (e.g., 'happy', 'sad', 'calm', 'angry').
    """
    emotion = emotion.lower().strip()
    
    if emotion not in EMOTION_PLAYLISTS:
        logger.warning(f"Emotion '{emotion}' is not mapped to any playlist.")
        print(f"Available emotions are: {', '.join(EMOTION_PLAYLISTS.keys())}")
        return

    playlist_id = EMOTION_PLAYLISTS[emotion]
    playlist_url = get_playlist_url(playlist_id)
    
    print(f"\n>>> Opening {emotion.upper()} playlist...")
    print(f">>> URL: {playlist_url}")
    
    try:
        webbrowser.open(playlist_url)
        logger.info(f"Successfully triggered browser to open {emotion} playlist.")
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

if __name__ == "__main__":
    print("--- Spotify Emotion-Aware Player Test ---")
    available_emotions = ", ".join(EMOTION_PLAYLISTS.keys())
    print(f"Mapped emotions: {available_emotions}")
    
    user_emotion = input("\nEnter an emotion to play music: ").strip()
    open_playlist_for_emotion(user_emotion)
