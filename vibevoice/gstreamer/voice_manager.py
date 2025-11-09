"""
Voice Manager

Manages voice reference audio files for VibeVoice TTS.
"""
import os
from typing import Dict, Optional


class VoiceManager:
    """Manages voice reference audio files"""

    def __init__(self):
        self.voices: Dict[str, str] = {}  # name -> audio_path
        self.current_voice = None

    def load_voice(self, name: str, audio_path: str, element=None):
        """Load a voice reference"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Voice audio not found: {audio_path}")

        self.voices[name] = audio_path
        print(f"Voice '{name}' loaded from {audio_path}")
        if element:
            element.emit("voice-loaded", name)

    def unload_voice(self, name: str):
        """Unload a voice reference"""
        if name in self.voices:
            del self.voices[name]
            print(f"Voice '{name}' unloaded")

            if self.current_voice == name:
                self.current_voice = None

    def set_current_voice(self, name: str, element=None):
        """Set the active voice"""
        if name not in self.voices:
            raise ValueError(f"Voice '{name}' not loaded")

        self.current_voice = name
        print(f"Current voice set to '{name}'")
        if element:
            element.emit("voice-changed", name)

    def get_current_voice_path(self) -> Optional[str]:
        """Get the current voice audio path"""
        if self.current_voice is None:
            return None
        return self.voices.get(self.current_voice)

    def list_voices(self):
        """List all loaded voices"""
        return list(self.voices.keys())

    def clear(self):
        """Clear all voices"""
        self.voices.clear()
        self.current_voice = None
