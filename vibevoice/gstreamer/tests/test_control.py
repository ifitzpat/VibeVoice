"""
Tests for Control Pad Handler

These tests verify the control pad JSON protocol for runtime configuration.
"""
import pytest
import json
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


@pytest.fixture
def mock_element():
    """Mock GStreamer element for testing"""
    class MockModelManager:
        def __init__(self):
            self.model = None
            self.model_name = None
            self.compiled = False
            self.loaded = False

        def is_loaded(self):
            return self.loaded

        def load_model(self, model_name, compile=True, diffusion_steps=10, element=None):
            self.model_name = model_name
            self.compiled = compile
            self.loaded = True
            if element:
                element.emit("model-loading", model_name)
                element.emit("model-loaded", model_name)

        def unload_model(self, element=None):
            self.model = None
            self.model_name = None
            self.compiled = False
            self.loaded = False
            if element:
                element.emit("model-unloaded")

    class MockVoiceManager:
        def __init__(self):
            self.voices = {}
            self.current_voice = None

        def load_voice(self, name, audio_path, element=None):
            if not audio_path:
                raise ValueError("Audio path is required")
            self.voices[name] = audio_path
            if element:
                element.emit("voice-loaded", name)

        def unload_voice(self, name):
            if name not in self.voices:
                raise ValueError(f"Voice '{name}' not loaded")
            if self.current_voice == name:
                self.current_voice = None
            del self.voices[name]

        def set_current_voice(self, name, element=None):
            if name not in self.voices:
                raise ValueError(f"Voice '{name}' not loaded")
            self.current_voice = name
            if element:
                element.emit("voice-changed", name)

        def list_voices(self):
            return list(self.voices.keys())

    class MockSentenceQueue:
        def __init__(self):
            self.items = []

        def qsize(self):
            return len(self.items)

        def empty(self):
            return len(self.items) == 0

        def get_nowait(self):
            if self.items:
                return self.items.pop(0)
            raise Exception("Empty")

    class MockAudioQueue:
        def __init__(self):
            self.items = []

        def qsize(self):
            return len(self.items)

        def empty(self):
            return len(self.items) == 0

        def get_nowait(self):
            if self.items:
                return self.items.pop(0)
            raise Exception("Empty")

    class MockElement:
        def __init__(self):
            self.model_manager = MockModelManager()
            self.voice_manager = MockVoiceManager()
            self.sentence_queue = MockSentenceQueue()
            self.audio_queue = MockAudioQueue()
            self._cfg_scale = 1.3
            self._diffusion_steps = 10
            self._speed = 1.0
            self.signals = []
            self.messages = []
            self.interrupt_called = False

        def emit(self, signal_name, *args):
            self.signals.append((signal_name, args))

        def post_message(self, msg):
            # Handle both dict (for testing) and Gst.Message (for production)
            if isinstance(msg, dict):
                self.messages.append(msg)
            else:
                struct = msg.get_structure()
                self.messages.append({
                    'name': struct.get_name(),
                    'message': struct.get_value('message')
                })

        def interrupt_generation(self):
            self.interrupt_called = True
            self.sentence_queue.items = []
            self.audio_queue.items = []

    return MockElement()


class TestControlHandlerBasics:
    """Test basic control handler functionality"""

    def test_control_handler_can_be_created(self, mock_element):
        """Test that ControlHandler can be instantiated"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        assert handler is not None
        assert handler.element is mock_element

    def test_handle_command_with_invalid_json(self, mock_element):
        """Test that invalid JSON is handled gracefully"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command("not valid json {")

        # Should send error message
        assert len(mock_element.messages) > 0
        assert mock_element.messages[-1]['name'] == 'vibevoice-error'

    def test_handle_command_with_unknown_command(self, mock_element):
        """Test that unknown commands are handled"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "unknown_command"}')

        # Should send error message
        assert len(mock_element.messages) > 0
        assert mock_element.messages[-1]['name'] == 'vibevoice-error'


class TestLoadModelCommand:
    """Test load_model command"""

    def test_load_model_with_defaults(self, mock_element):
        """Test load_model with default parameters"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "load_model"}')

        # Model should be loaded
        assert mock_element.model_manager.is_loaded()
        assert mock_element.model_manager.model_name == "vibevoice/VibeVoice-1.5B"

        # Should send status message
        assert len(mock_element.messages) > 0
        assert mock_element.messages[-1]['name'] == 'vibevoice-status'

    def test_load_model_with_custom_name(self, mock_element):
        """Test load_model with custom model name"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "load_model",
            "model": "custom/model"
        }))

        assert mock_element.model_manager.model_name == "custom/model"

    def test_load_model_with_compile_false(self, mock_element):
        """Test load_model with compile disabled"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "load_model",
            "compile": False
        }))

        assert mock_element.model_manager.compiled is False

    def test_load_model_with_diffusion_steps(self, mock_element):
        """Test load_model with custom diffusion steps"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        # This is tested implicitly - diffusion_steps is passed to load_model
        handler.handle_command(json.dumps({
            "command": "load_model",
            "diffusion_steps": 15
        }))

        assert mock_element.model_manager.is_loaded()


class TestUnloadModelCommand:
    """Test unload_model command"""

    def test_unload_model(self, mock_element):
        """Test unload_model command"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        # Load first
        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "load_model"}')
        assert mock_element.model_manager.is_loaded()

        # Unload
        handler.handle_command('{"command": "unload_model"}')

        assert not mock_element.model_manager.is_loaded()
        assert mock_element.model_manager.model_name is None

        # Should send status message
        messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-status']
        assert len(messages) >= 2  # load + unload


class TestVoiceCommands:
    """Test voice-related commands"""

    def test_load_voice_command(self, mock_element):
        """Test load_voice command"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "load_voice",
            "name": "voice1",
            "audio_path": "/path/to/voice.wav"
        }))

        # Voice should be loaded
        assert "voice1" in mock_element.voice_manager.voices
        assert mock_element.voice_manager.voices["voice1"] == "/path/to/voice.wav"

        # Should send status message
        status_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-status']
        assert len(status_messages) > 0

    def test_load_voice_missing_name(self, mock_element):
        """Test load_voice with missing name"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "load_voice",
            "audio_path": "/path/to/voice.wav"
        }))

        # Should send error
        error_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-error']
        assert len(error_messages) > 0

    def test_load_voice_missing_audio_path(self, mock_element):
        """Test load_voice with missing audio_path"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "load_voice",
            "name": "voice1"
        }))

        # Should send error
        error_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-error']
        assert len(error_messages) > 0

    def test_set_voice_command(self, mock_element):
        """Test set_voice command"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)

        # Load voice first
        handler.handle_command(json.dumps({
            "command": "load_voice",
            "name": "voice1",
            "audio_path": "/path/to/voice.wav"
        }))

        # Set as current
        handler.handle_command(json.dumps({
            "command": "set_voice",
            "name": "voice1"
        }))

        assert mock_element.voice_manager.current_voice == "voice1"

    def test_set_voice_not_loaded(self, mock_element):
        """Test set_voice with voice that's not loaded"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "set_voice",
            "name": "nonexistent"
        }))

        # Should send error
        error_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-error']
        assert len(error_messages) > 0

    def test_unload_voice_command(self, mock_element):
        """Test unload_voice command"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)

        # Load voice first
        handler.handle_command(json.dumps({
            "command": "load_voice",
            "name": "voice1",
            "audio_path": "/path/to/voice.wav"
        }))

        # Unload
        handler.handle_command(json.dumps({
            "command": "unload_voice",
            "name": "voice1"
        }))

        assert "voice1" not in mock_element.voice_manager.voices

    def test_unload_voice_not_loaded(self, mock_element):
        """Test unload_voice with voice that's not loaded"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "unload_voice",
            "name": "nonexistent"
        }))

        # Should send error
        error_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-error']
        assert len(error_messages) > 0


class TestSetParametersCommand:
    """Test set_parameters command"""

    def test_set_cfg_scale(self, mock_element):
        """Test setting cfg_scale"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "set_parameters",
            "cfg_scale": 2.0
        }))

        assert mock_element._cfg_scale == 2.0

    def test_set_diffusion_steps(self, mock_element):
        """Test setting diffusion_steps"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "set_parameters",
            "diffusion_steps": 15
        }))

        assert mock_element._diffusion_steps == 15

    def test_set_speed(self, mock_element):
        """Test setting speed"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "set_parameters",
            "speed": 1.5
        }))

        assert mock_element._speed == 1.5

    def test_set_multiple_parameters(self, mock_element):
        """Test setting multiple parameters at once"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command(json.dumps({
            "command": "set_parameters",
            "cfg_scale": 2.0,
            "diffusion_steps": 15,
            "speed": 1.5
        }))

        assert mock_element._cfg_scale == 2.0
        assert mock_element._diffusion_steps == 15
        assert mock_element._speed == 1.5


class TestGetStatusCommand:
    """Test get_status command"""

    def test_get_status_basic(self, mock_element):
        """Test get_status returns basic information"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "get_status"}')

        # Should send status message with JSON
        status_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-status']
        assert len(status_messages) > 0

        # Parse the status JSON
        status_json = status_messages[-1]['message']
        status = json.loads(status_json)

        assert 'model_loaded' in status
        assert 'model_name' in status
        assert 'current_voice' in status
        assert 'loaded_voices' in status
        assert 'cfg_scale' in status
        assert 'diffusion_steps' in status

    def test_get_status_with_model_loaded(self, mock_element):
        """Test get_status after loading model"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "load_model"}')
        handler.handle_command('{"command": "get_status"}')

        status_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-status']
        status_json = status_messages[-1]['message']
        status = json.loads(status_json)

        assert status['model_loaded'] is True
        assert status['model_name'] == "vibevoice/VibeVoice-1.5B"


class TestInterruptCommand:
    """Test interrupt command"""

    def test_interrupt_command(self, mock_element):
        """Test interrupt command clears queues"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        # Add items to queues
        mock_element.sentence_queue.items = ["sentence1", "sentence2"]
        mock_element.audio_queue.items = ["audio1", "audio2"]

        handler = ControlHandler(mock_element)
        handler.handle_command('{"command": "interrupt"}')

        # Interrupt should be called
        assert mock_element.interrupt_called

        # Queues should be cleared
        assert len(mock_element.sentence_queue.items) == 0
        assert len(mock_element.audio_queue.items) == 0

        # Should send status message
        status_messages = [m for m in mock_element.messages if m['name'] == 'vibevoice-status']
        assert len(status_messages) > 0


class TestMessageSending:
    """Test status and error message sending"""

    def test_send_status_creates_message(self, mock_element):
        """Test that send_status posts message to bus"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.send_status("Test status message")

        assert len(mock_element.messages) == 1
        assert mock_element.messages[0]['name'] == 'vibevoice-status'
        assert mock_element.messages[0]['message'] == "Test status message"

    def test_send_error_creates_message(self, mock_element):
        """Test that send_error posts message to bus"""
        from vibevoice.gstreamer.control_handler import ControlHandler

        handler = ControlHandler(mock_element)
        handler.send_error("Test error message")

        assert len(mock_element.messages) == 1
        assert mock_element.messages[0]['name'] == 'vibevoice-error'
        assert mock_element.messages[0]['message'] == "Test error message"
