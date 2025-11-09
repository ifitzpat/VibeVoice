"""
Tests for GstVibeVoice element

These tests verify the element's basic functionality including pads,
properties, and state management.
"""
import pytest
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


@pytest.fixture
def vibevoice_element():
    """Create a VibeVoice element for testing"""
    from vibevoice.gstreamer.plugin import register_plugin

    register_plugin()
    element = Gst.ElementFactory.make("vibevoice", "test")
    return element


class TestElementPads:
    """Test element pad configuration"""

    def test_has_sink_pad(self, vibevoice_element):
        """Test that element has a sink pad for text input"""
        sinkpad = vibevoice_element.get_static_pad("sink")
        assert sinkpad is not None
        assert sinkpad.get_direction() == Gst.PadDirection.SINK

    def test_has_src_pad(self, vibevoice_element):
        """Test that element has a src pad for audio output"""
        srcpad = vibevoice_element.get_static_pad("src")
        assert srcpad is not None
        assert srcpad.get_direction() == Gst.PadDirection.SRC

    def test_sink_pad_accepts_text(self, vibevoice_element):
        """Test that sink pad accepts text capabilities"""
        sinkpad = vibevoice_element.get_static_pad("sink")
        caps = sinkpad.get_pad_template_caps()

        # Should accept text/x-raw
        assert caps.can_intersect(Gst.Caps.from_string("text/x-raw"))

    def test_src_pad_produces_audio(self, vibevoice_element):
        """Test that src pad produces audio capabilities"""
        srcpad = vibevoice_element.get_static_pad("src")
        caps = srcpad.get_pad_template_caps()

        # Should produce audio/x-raw at 24000Hz
        test_caps = Gst.Caps.from_string(
            "audio/x-raw,format=F32LE,rate=24000,channels=1"
        )
        assert caps.can_intersect(test_caps)

    def test_control_pad_can_be_requested(self, vibevoice_element):
        """Test that control pad can be requested"""
        # Control pad should not exist initially
        controlpad = vibevoice_element.get_static_pad("control")
        assert controlpad is None

        # Request control pad
        template = vibevoice_element.get_pad_template("control")
        assert template is not None

        controlpad = vibevoice_element.request_pad(template, "control", None)
        assert controlpad is not None
        assert controlpad.get_direction() == Gst.PadDirection.SINK


class TestElementProperties:
    """Test element properties"""

    def test_model_property_exists(self, vibevoice_element):
        """Test that model property exists and has default value"""
        model = vibevoice_element.get_property("model")
        assert model is not None
        assert "VibeVoice" in model

    def test_model_property_is_writable(self, vibevoice_element):
        """Test that model property can be set"""
        vibevoice_element.set_property("model", "vibevoice/VibeVoice-1.5B")
        assert vibevoice_element.get_property("model") == "vibevoice/VibeVoice-1.5B"

    def test_cfg_scale_property(self, vibevoice_element):
        """Test CFG scale property"""
        # Default value
        assert vibevoice_element.get_property("cfg-scale") == 1.3

        # Set new value
        vibevoice_element.set_property("cfg-scale", 1.5)
        assert vibevoice_element.get_property("cfg-scale") == 1.5

    def test_diffusion_steps_property(self, vibevoice_element):
        """Test diffusion steps property"""
        # Default value
        assert vibevoice_element.get_property("diffusion-steps") == 10

        # Set new value
        vibevoice_element.set_property("diffusion-steps", 15)
        assert vibevoice_element.get_property("diffusion-steps") == 15

    def test_compile_property(self, vibevoice_element):
        """Test compile property"""
        # Default should be True
        assert vibevoice_element.get_property("compile") is True

        # Can be disabled
        vibevoice_element.set_property("compile", False)
        assert vibevoice_element.get_property("compile") is False

    def test_sentence_queue_size_property(self, vibevoice_element):
        """Test sentence queue size property"""
        # Default should be 100
        assert vibevoice_element.get_property("sentence-queue-size") == 100

        # Can be changed
        vibevoice_element.set_property("sentence-queue-size", 50)
        assert vibevoice_element.get_property("sentence-queue-size") == 50


class TestElementSignals:
    """Test element signals"""

    def test_model_loading_signal(self, vibevoice_element):
        """Test that model-loading signal can be connected"""
        signal_received = []

        def on_model_loading(element, model_name):
            signal_received.append(model_name)

        vibevoice_element.connect("model-loading", on_model_loading)

        # Emit signal manually for testing
        vibevoice_element.emit("model-loading", "test-model")

        assert len(signal_received) == 1
        assert signal_received[0] == "test-model"

    def test_ready_signal(self, vibevoice_element):
        """Test that ready signal can be connected"""
        signal_received = []

        def on_ready(element, is_ready):
            signal_received.append(is_ready)

        vibevoice_element.connect("ready", on_ready)
        vibevoice_element.emit("ready", True)

        assert len(signal_received) == 1
        assert signal_received[0] is True

    def test_interrupted_signal(self, vibevoice_element):
        """Test that interrupted signal can be connected"""
        signal_received = []

        def on_interrupted(element):
            signal_received.append(True)

        vibevoice_element.connect("interrupted", on_interrupted)
        vibevoice_element.emit("interrupted")

        assert len(signal_received) == 1


class TestElementStateChanges:
    """Test element state transitions"""

    def test_null_to_ready(self, vibevoice_element):
        """Test NULL to READY state transition"""
        ret = vibevoice_element.set_state(Gst.State.READY)
        assert ret == Gst.StateChangeReturn.SUCCESS

        ret, state, pending = vibevoice_element.get_state(Gst.CLOCK_TIME_NONE)
        assert state == Gst.State.READY

    def test_ready_to_paused(self, vibevoice_element):
        """Test READY to PAUSED state transition"""
        vibevoice_element.set_state(Gst.State.READY)
        ret = vibevoice_element.set_state(Gst.State.PAUSED)
        assert ret in [Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC]

    def test_paused_to_playing(self, vibevoice_element):
        """Test PAUSED to PLAYING state transition"""
        vibevoice_element.set_state(Gst.State.PAUSED)
        ret = vibevoice_element.set_state(Gst.State.PLAYING)
        assert ret in [Gst.StateChangeReturn.SUCCESS, Gst.StateChangeReturn.ASYNC]

    def test_cleanup_on_null(self, vibevoice_element):
        """Test that element cleans up when going to NULL state"""
        vibevoice_element.set_state(Gst.State.PLAYING)
        vibevoice_element.set_state(Gst.State.NULL)

        ret, state, pending = vibevoice_element.get_state(Gst.CLOCK_TIME_NONE)
        assert state == Gst.State.NULL
