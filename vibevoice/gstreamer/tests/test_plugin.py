"""
Tests for GStreamer plugin registration

These tests verify that the plugin can be registered and discovered by GStreamer.
"""
import pytest
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


class TestPluginRegistration:
    """Test plugin registration and discovery"""

    def test_gst_initialized(self):
        """Test that GStreamer is initialized"""
        assert Gst.is_initialized()

    def test_plugin_can_be_registered(self):
        """Test that we can register the VibeVoice plugin"""
        from vibevoice.gstreamer.element import GstVibeVoice
        from gi.repository import GObject

        # Ensure type can be registered
        gtype = GObject.type_register(GstVibeVoice)
        assert gtype is not None

    def test_element_can_be_instantiated_directly(self):
        """Test that we can instantiate the element directly"""
        # Python GStreamer elements can be instantiated directly
        # without factory registration
        from vibevoice.gstreamer.element import GstVibeVoice
        from gi.repository import GObject

        GObject.type_register(GstVibeVoice)
        element = GstVibeVoice()
        assert element is not None

    def test_element_has_required_pads(self):
        """Test that element has the required pads"""
        from vibevoice.gstreamer.element import GstVibeVoice
        from gi.repository import GObject

        GObject.type_register(GstVibeVoice)
        element = GstVibeVoice()

        # Check pads exist
        assert element.get_static_pad("sink") is not None
        assert element.get_static_pad("src") is not None

    def test_element_has_metadata(self):
        """Test that element class has metadata defined"""
        from vibevoice.gstreamer.element import GstVibeVoice

        # Check __gstmetadata__ exists and is correct
        assert hasattr(GstVibeVoice, '__gstmetadata__')
        metadata = GstVibeVoice.__gstmetadata__
        assert len(metadata) == 4
        assert "VibeVoice" in metadata[0]
        assert "TTS" in metadata[1]
