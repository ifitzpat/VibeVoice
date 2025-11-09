"""
GStreamer Plugin Registration

Registers the VibeVoice TTS element with GStreamer.
"""
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

from .element import GstVibeVoice


# Global flag to track if plugin is registered
_plugin_registered = False


def register_plugin():
    """
    Register the VibeVoice element with GStreamer.

    This allows the element to be created using Gst.ElementFactory.make("vibevoice")
    """
    global _plugin_registered

    if _plugin_registered:
        return True

    # Ensure GStreamer is initialized
    if not Gst.is_initialized():
        Gst.init(None)

    # Register the element type with GObject
    gtype = GObject.type_register(GstVibeVoice)

    # For Python GStreamer elements, we need to ensure the element class is properly initialized
    # The __gstmetadata__ attribute should be picked up automatically

    # Register with GStreamer element factory
    success = Gst.Element.register(
        None,          # Plugin (None for static registration)
        "vibevoice",   # Element name
        Gst.Rank.PRIMARY,  # Higher rank so it can be found
        gtype          # GType
    )

    if success:
        _plugin_registered = True
        print("VibeVoice plugin registered successfully")
    else:
        print("Failed to register VibeVoice plugin")

    return success
