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
        # This will be implemented when we create the plugin
        # For now, we'll test that we can register a type
        from vibevoice.gstreamer.element import GstVibeVoice

        # Type should be registered
        assert hasattr(GstVibeVoice, '__gtype__')

    def test_element_factory_exists(self):
        """Test that element factory is available after registration"""
        # Try to get the element factory
        # This requires the plugin to be registered with GStreamer
        from vibevoice.gstreamer.plugin import register_plugin

        # Register plugin
        register_plugin()

        # Check if element factory exists
        factory = Gst.ElementFactory.find("vibevoice")
        assert factory is not None

    def test_element_can_be_created(self):
        """Test that we can create a VibeVoice element"""
        from vibevoice.gstreamer.plugin import register_plugin

        register_plugin()
        element = Gst.ElementFactory.make("vibevoice", "test_tts")
        assert element is not None
        assert element.get_name() == "test_tts"

    def test_element_metadata(self):
        """Test that element has correct metadata"""
        from vibevoice.gstreamer.plugin import register_plugin

        register_plugin()
        factory = Gst.ElementFactory.find("vibevoice")

        # Check metadata
        assert "VibeVoice" in factory.get_metadata("long-name")
        assert "TTS" in factory.get_metadata("klass")
