#!/usr/bin/env python3
"""
Basic VibeVoice GStreamer Plugin Usage Example

This example demonstrates the simplest way to use the VibeVoice GStreamer element
to convert text to speech and play it through your audio output.
"""
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def on_message(bus, message):
    """Handle GStreamer bus messages"""
    t = message.type

    if t == Gst.MessageType.EOS:
        print("End of stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err}, {debug}")
        loop.quit()
    elif t == Gst.MessageType.APPLICATION:
        struct = message.get_structure()
        if struct.get_name() == "vibevoice-status":
            status = struct.get_value("message")
            print(f"Status: {status}")
        elif struct.get_name() == "vibevoice-error":
            error = struct.get_value("message")
            print(f"VibeVoice Error: {error}")

    return True

def on_ready(vibevoice, ready):
    """Handle ready signal - model is loaded and ready to generate"""
    if ready:
        print("Model ready! Generating speech...")

def on_model_loaded(vibevoice, model_name):
    """Handle model-loaded signal"""
    print(f"Model loaded: {model_name}")

def on_generation_complete(vibevoice):
    """Handle generation-complete signal"""
    print("Generation complete!")

# Create pipeline
# vibevoice element converts text to audio
# audioconvert ensures proper format
# autoaudiosink automatically selects audio output
pipeline = Gst.parse_launch(
    "vibevoice name=tts ! audioconvert ! autoaudiosink"
)

# Get the vibevoice element
vibevoice = pipeline.get_by_name("tts")

# Set properties
vibevoice.set_property("model-name", "vibevoice/VibeVoice-1.5B")
vibevoice.set_property("compile", True)  # Use torch.compile for faster inference
vibevoice.set_property("cfg-scale", 1.3)
vibevoice.set_property("diffusion-steps", 10)

# Connect to signals
vibevoice.connect("ready", on_ready)
vibevoice.connect("model-loaded", on_model_loaded)
vibevoice.connect("generation-complete", on_generation_complete)

# Set up bus message handling
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Start pipeline
print("Starting pipeline...")
pipeline.set_state(Gst.State.PLAYING)

# Send text to the sink pad
print("Sending text...")
sinkpad = vibevoice.get_static_pad("sink")
text = "Hello! This is a test of the VibeVoice GStreamer plugin."
buffer = Gst.Buffer.new_wrapped(text.encode('utf-8'))
sinkpad.chain(buffer)

# Send EOS to signal end of text
eos_event = Gst.Event.new_eos()
sinkpad.send_event(eos_event)

# Run main loop
print("Playing audio...")
loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("Interrupted")

# Cleanup
pipeline.set_state(Gst.State.NULL)
print("Done!")
