import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spotify_player import open_playlist_for_emotion

print("--- Automated Verification of Spotify Player ---")

print("\n[TEST 1] Valid Emotion: 'happy'")
open_playlist_for_emotion("happy")

print("\n[TEST 2] Valid Emotion (case-sensitivity check): 'SAD'")
open_playlist_for_emotion("SAD")

print("\n[TEST 3] Invalid Emotion: 'confused'")
open_playlist_for_emotion("confused")

print("\nVerification script finished. (Browser windows should have opened for TEST 1 and 2)")
