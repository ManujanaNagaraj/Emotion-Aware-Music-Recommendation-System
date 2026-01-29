"""
Spotify Configuration Module

This module serves as a centralized configuration for mapping detected emotion labels
to Spotify playlist IDs. This allows for clean separation between the emotion
recognition logic and the music recommendation experience.

Configuration instructions:
1. Playlist IDs can be found in the Spotify Share URL (e.g., https://open.spotify.com/playlist/37i9dQZF1DXdPec7WLTmlr?...)
   The ID is the alphanumeric string between 'playlist/' and the '?'.
2. Replace the placeholder IDs in the `EMOTION_PLAYLISTS` dictionary with your preferred IDs.
"""

from typing import Dict

# Dictionary mapping emotion labels to Spotify playlist IDs.
# Emotion labels must match those defined in src/emotion_config.py.
# Replace these placeholders with actual Spotify playlist IDs.
EMOTION_PLAYLISTS: Dict[str, str] = {
    "happy": "14iO0n19NQKgEI0ibuofeV",  # Example: '37i9dQZF1DXdPec7WLTmlr'
    "sad": "0AyOLKzLZZmlliok7bu1mp",      # Example: '37i9dQZF1DX3YSRY7tZ9b3'
    "calm": "0TOng1FTBaa6bHJcfack1S",    # Example: '37i9dQZF1DX4sW36Cj2m2y'
    "angry": "4Ky4vveGq77DAgV3Z2Lk4e"   # Example: '37i9dQZF1DX3RxPs9vCghT'
}

def get_playlist_url(playlist_id: str) -> str:
    """
    Converts a Spotify playlist ID into a full web URL.

    Playlist IDs are used instead of full URLs to:
    1. Maintain consistency (API interactions usually require IDs).
    2. Keep the configuration cleaner and more robust to URL format changes.
    3. Allow for programmatic construction of various Spotify integration types 
       (e.g., embed players, API calls, or web links).

    Args:
        playlist_id (str): The Spotify playlist ID.

    Returns:
        str: The full web URL for the playlist.
    """
    return f"https://open.spotify.com/playlist/{playlist_id}"

if __name__ == "__main__":
    # Internal validation/demo logic
    print("--- Spotify Configuration Summary ---")
    for emotion, pl_id in EMOTION_PLAYLISTS.items():
        url = get_playlist_url(pl_id)
        print(f"Emotion: {emotion:6} | ID: {pl_id:30} | URL: {url}")
