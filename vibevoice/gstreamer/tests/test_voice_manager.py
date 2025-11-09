"""
Tests for VoiceManager

These tests verify voice loading, unloading, and management.
"""
import pytest
import os
import tempfile


@pytest.fixture
def voice_manager():
    """Create a VoiceManager instance for testing"""
    from vibevoice.gstreamer.voice_manager import VoiceManager

    return VoiceManager()


@pytest.fixture
def temp_audio_file():
    """Create a temporary audio file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"fake audio data")
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestVoiceManagerBasics:
    """Test basic VoiceManager functionality"""

    def test_voice_manager_initialization(self, voice_manager):
        """Test that VoiceManager initializes correctly"""
        assert voice_manager.voices == {}
        assert voice_manager.current_voice is None

    def test_list_voices_empty_initially(self, voice_manager):
        """Test that list_voices returns empty list initially"""
        assert voice_manager.list_voices() == []

    def test_get_current_voice_path_returns_none_initially(self, voice_manager):
        """Test that get_current_voice_path returns None when no voice is set"""
        assert voice_manager.get_current_voice_path() is None


class TestVoiceLoading:
    """Test voice loading functionality"""

    def test_load_voice_adds_to_dictionary(self, voice_manager, temp_audio_file):
        """Test that load_voice adds voice to voices dictionary"""
        voice_manager.load_voice("Alice", temp_audio_file)

        assert "Alice" in voice_manager.voices
        assert voice_manager.voices["Alice"] == temp_audio_file

    def test_load_voice_shows_in_list(self, voice_manager, temp_audio_file):
        """Test that loaded voice appears in list_voices"""
        voice_manager.load_voice("Alice", temp_audio_file)

        voices = voice_manager.list_voices()
        assert "Alice" in voices

    def test_load_voice_raises_if_file_not_found(self, voice_manager):
        """Test that load_voice raises FileNotFoundError for non-existent file"""
        with pytest.raises(FileNotFoundError):
            voice_manager.load_voice("Alice", "/nonexistent/path.wav")

    def test_load_voice_emits_signal(self, voice_manager, temp_audio_file):
        """Test that load_voice emits voice-loaded signal"""
        class MockElement:
            def __init__(self):
                self.signals = []

            def emit(self, signal_name, *args):
                self.signals.append((signal_name, args))

        element = MockElement()
        voice_manager.load_voice("Alice", temp_audio_file, element=element)

        signal_names = [s[0] for s in element.signals]
        assert "voice-loaded" in signal_names

        # Check the signal has the voice name
        voice_loaded = [s for s in element.signals if s[0] == "voice-loaded"][0]
        assert voice_loaded[1] == ("Alice",)

    def test_load_multiple_voices(self, voice_manager, temp_audio_file):
        """Test loading multiple voices"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.load_voice("Bob", temp_audio_file)
        voice_manager.load_voice("Charlie", temp_audio_file)

        assert len(voice_manager.list_voices()) == 3
        assert "Alice" in voice_manager.voices
        assert "Bob" in voice_manager.voices
        assert "Charlie" in voice_manager.voices


class TestVoiceUnloading:
    """Test voice unloading functionality"""

    def test_unload_voice_removes_from_dictionary(self, voice_manager, temp_audio_file):
        """Test that unload_voice removes voice from dictionary"""
        voice_manager.load_voice("Alice", temp_audio_file)
        assert "Alice" in voice_manager.voices

        voice_manager.unload_voice("Alice")
        assert "Alice" not in voice_manager.voices

    def test_unload_voice_clears_current_if_active(self, voice_manager, temp_audio_file):
        """Test that unloading current voice clears current_voice"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.set_current_voice("Alice")
        assert voice_manager.current_voice == "Alice"

        voice_manager.unload_voice("Alice")
        assert voice_manager.current_voice is None

    def test_unload_voice_keeps_current_if_different(self, voice_manager, temp_audio_file):
        """Test that unloading non-current voice keeps current_voice"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.load_voice("Bob", temp_audio_file)
        voice_manager.set_current_voice("Alice")

        voice_manager.unload_voice("Bob")
        assert voice_manager.current_voice == "Alice"

    def test_unload_voice_safe_if_not_loaded(self, voice_manager):
        """Test that unload_voice is safe when voice doesn't exist"""
        # Should not raise an error
        voice_manager.unload_voice("NonExistent")

    def test_clear_removes_all_voices(self, voice_manager, temp_audio_file):
        """Test that clear removes all voices"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.load_voice("Bob", temp_audio_file)
        voice_manager.set_current_voice("Alice")

        voice_manager.clear()

        assert voice_manager.voices == {}
        assert voice_manager.current_voice is None
        assert voice_manager.list_voices() == []


class TestCurrentVoice:
    """Test current voice functionality"""

    def test_set_current_voice_sets_current(self, voice_manager, temp_audio_file):
        """Test that set_current_voice sets the current voice"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.set_current_voice("Alice")

        assert voice_manager.current_voice == "Alice"

    def test_set_current_voice_raises_if_not_loaded(self, voice_manager):
        """Test that set_current_voice raises if voice not loaded"""
        with pytest.raises(ValueError, match="not loaded"):
            voice_manager.set_current_voice("NonExistent")

    def test_set_current_voice_emits_signal(self, voice_manager, temp_audio_file):
        """Test that set_current_voice emits voice-changed signal"""
        voice_manager.load_voice("Alice", temp_audio_file)

        class MockElement:
            def __init__(self):
                self.signals = []

            def emit(self, signal_name, *args):
                self.signals.append((signal_name, args))

        element = MockElement()
        voice_manager.set_current_voice("Alice", element=element)

        signal_names = [s[0] for s in element.signals]
        assert "voice-changed" in signal_names

        voice_changed = [s for s in element.signals if s[0] == "voice-changed"][0]
        assert voice_changed[1] == ("Alice",)

    def test_get_current_voice_path_returns_path(self, voice_manager, temp_audio_file):
        """Test that get_current_voice_path returns the correct path"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.set_current_voice("Alice")

        path = voice_manager.get_current_voice_path()
        assert path == temp_audio_file

    def test_get_current_voice_path_returns_none_if_not_set(self, voice_manager, temp_audio_file):
        """Test that get_current_voice_path returns None when no voice is set"""
        voice_manager.load_voice("Alice", temp_audio_file)
        # Don't set current voice

        assert voice_manager.get_current_voice_path() is None

    def test_switch_current_voice(self, voice_manager, temp_audio_file):
        """Test switching between voices"""
        voice_manager.load_voice("Alice", temp_audio_file)
        voice_manager.load_voice("Bob", temp_audio_file)

        voice_manager.set_current_voice("Alice")
        assert voice_manager.current_voice == "Alice"

        voice_manager.set_current_voice("Bob")
        assert voice_manager.current_voice == "Bob"
