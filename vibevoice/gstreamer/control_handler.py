"""
Control Handler for VibeVoice GStreamer Element

Handles JSON control messages for runtime configuration.
"""
import json
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class ControlHandler:
    """
    Handles control pad messages for VibeVoice element.

    Processes JSON commands for:
    - Model loading/unloading
    - Voice management
    - Parameter adjustment
    - Status queries
    - Generation interruption
    """

    def __init__(self, element):
        """
        Initialize ControlHandler.

        Args:
            element: Parent GstVibeVoice element
        """
        self.element = element

    def handle_command(self, command_json: str):
        """
        Process a control command.

        Args:
            command_json: JSON string containing command
        """
        try:
            command = json.loads(command_json)
        except json.JSONDecodeError as e:
            # Invalid JSON in control message
            self.send_error(f"Invalid JSON: {e}")
            return

        cmd_type = command.get("command")

        if cmd_type == "load_model":
            self.handle_load_model(command)
        elif cmd_type == "unload_model":
            self.handle_unload_model()
        elif cmd_type == "load_voice":
            self.handle_load_voice(command)
        elif cmd_type == "unload_voice":
            self.handle_unload_voice(command)
        elif cmd_type == "set_voice":
            self.handle_set_voice(command)
        elif cmd_type == "set_parameters":
            self.handle_set_parameters(command)
        elif cmd_type == "get_status":
            self.handle_get_status()
        elif cmd_type == "interrupt":
            self.handle_interrupt()
        else:
            # Unknown command
            self.send_error(f"Unknown command: {cmd_type}")

    def handle_load_model(self, command):
        """
        Handle load_model command.

        Args:
            command: Command dict with optional model, compile, diffusion_steps
        """
        model_name = command.get("model", "vibevoice/VibeVoice-1.5B")
        compile = command.get("compile", True)
        diffusion_steps = command.get("diffusion_steps", 10)

        try:
            self.element.model_manager.load_model(
                model_name,
                compile=compile,
                diffusion_steps=diffusion_steps,
                element=self.element
            )
            self.send_status(f"Model loaded: {model_name}")
        except Exception as e:
            # Failed to load model
            self.send_error(f"Model load failed: {e}")
            self.element.emit("generation-error", str(e))

    def handle_unload_model(self):
        """Handle unload_model command."""
        try:
            self.element.model_manager.unload_model(element=self.element)
            self.send_status("Model unloaded")
        except Exception as e:
            # Failed to unload model
            self.send_error(f"Model unload failed: {e}")
            self.element.emit("generation-error", str(e))

    def handle_load_voice(self, command):
        """
        Handle load_voice command.

        Args:
            command: Command dict with name and audio_path
        """
        name = command.get("name")
        audio_path = command.get("audio_path")

        if not name or not audio_path:
            self.send_error("Missing 'name' or 'audio_path' in load_voice command")
            return

        try:
            self.element.voice_manager.load_voice(name, audio_path, element=self.element)
            self.send_status(f"Voice loaded: {name}")
        except Exception as e:
            # Failed to load voice
            self.send_error(f"Voice load failed: {e}")
            self.element.emit("generation-error", str(e))

    def handle_unload_voice(self, command):
        """
        Handle unload_voice command.

        Args:
            command: Command dict with name
        """
        name = command.get("name")

        if not name:
            self.send_error("Missing 'name' in unload_voice command")
            return

        try:
            self.element.voice_manager.unload_voice(name)
            self.send_status(f"Voice unloaded: {name}")
        except Exception as e:
            # Failed to unload voice
            self.send_error(f"Voice unload failed: {e}")

    def handle_set_voice(self, command):
        """
        Handle set_voice command.

        Args:
            command: Command dict with name
        """
        name = command.get("name")

        if not name:
            self.send_error("Missing 'name' in set_voice command")
            return

        try:
            self.element.voice_manager.set_current_voice(name, element=self.element)
            self.send_status(f"Current voice set to: {name}")
        except Exception as e:
            # Failed to set voice
            self.send_error(f"Set voice failed: {e}")
            self.element.emit("generation-error", str(e))

    def handle_set_parameters(self, command):
        """
        Handle set_parameters command.

        Args:
            command: Command dict with optional cfg_scale, diffusion_steps, speed
        """
        if "cfg_scale" in command:
            self.element._cfg_scale = command["cfg_scale"]

        if "diffusion_steps" in command:
            self.element._diffusion_steps = command["diffusion_steps"]
            # Note: Updating diffusion steps on loaded model would require
            # model.set_ddpm_inference_steps() - not implemented in mock

        if "speed" in command:
            self.element._speed = command["speed"]

        self.send_status("Parameters updated")

    def handle_get_status(self):
        """Handle get_status command - returns JSON status."""
        status = {
            "model_loaded": self.element.model_manager.is_loaded(),
            "model_name": self.element.model_manager.model_name,
            "compiled": self.element.model_manager.compiled,
            "current_voice": self.element.voice_manager.current_voice,
            "loaded_voices": self.element.voice_manager.list_voices(),
            "cfg_scale": self.element._cfg_scale,
            "diffusion_steps": self.element._diffusion_steps,
            "sentence_queue_size": self.element.sentence_queue.qsize(),
            "audio_queue_size": self.element.audio_queue.qsize(),
        }

        # Add VRAM usage if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                status["vram_usage_gb"] = torch.cuda.memory_allocated() / 1e9
        except ImportError:
            pass

        self.send_status(json.dumps(status, indent=2))

    def handle_interrupt(self):
        """Handle interrupt command - clear queues and interrupt generation."""
        try:
            self.element.interrupt_generation()
            self.send_status("Generation interrupted, buffers flushed")
        except Exception as e:
            # Failed to interrupt
            self.send_error(f"Interrupt failed: {e}")

    def send_status(self, message: str):
        """
        Send status message to bus.

        Args:
            message: Status message string
        """
        try:
            struct = Gst.Structure.new_empty("vibevoice-status")
            struct.set_value("message", message)

            msg = Gst.Message.new_application(self.element, struct)
            self.element.post_message(msg)
        except TypeError:
            # For testing: mock elements don't have proper GObject type
            # Call post_message directly with a dict
            self.element.post_message({
                'name': 'vibevoice-status',
                'message': message
            })

    def send_error(self, message: str):
        """
        Send error message to bus.

        Args:
            message: Error message string
        """
        try:
            struct = Gst.Structure.new_empty("vibevoice-error")
            struct.set_value("message", message)

            msg = Gst.Message.new_application(self.element, struct)
            self.element.post_message(msg)
        except TypeError:
            # For testing: mock elements don't have proper GObject type
            # Call post_message directly with a dict
            self.element.post_message({
                'name': 'vibevoice-error',
                'message': message
            })
