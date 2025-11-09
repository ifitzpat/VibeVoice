#!/usr/bin/env python3
"""
VibeVoice GStreamer Plugin - Control Pad Example

This example demonstrates how to use the control pad to dynamically
configure the VibeVoice element at runtime using JSON commands.

The control pad allows you to:
- Load/unload models
- Manage voice samples
- Adjust parameters on the fly
- Query status
- Interrupt generation
"""
import gi
import json
import time
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

def on_message(bus, message):
    """Handle GStreamer bus messages"""
    t = message.type

    if t == Gst.MessageType.EOS:
        print("\n=== End of stream ===")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"\nError: {err}, {debug}")
        loop.quit()
    elif t == Gst.MessageType.APPLICATION:
        struct = message.get_structure()
        if struct.get_name() == "vibevoice-status":
            status = struct.get_value("message")
            print(f"\n[STATUS] {status}")
        elif struct.get_name() == "vibevoice-error":
            error = struct.get_value("message")
            print(f"\n[ERROR] {error}")

    return True

def send_control_command(control_pad, command_dict):
    """Send a JSON control command to the control pad"""
    command_json = json.dumps(command_dict)
    print(f"\n>>> Sending command: {command_json}")

    buffer = Gst.Buffer.new_wrapped(command_json.encode('utf-8'))
    control_pad.chain(buffer)
    time.sleep(0.5)  # Give it time to process

# Create pipeline
pipeline = Gst.parse_launch(
    "vibevoice name=tts ! audioconvert ! autoaudiosink"
)

vibevoice = pipeline.get_by_name("tts")

# Get control pad
control_pad = vibevoice.get_static_pad("control")

# Set up bus message handling
bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)

# Start pipeline
print("Starting pipeline...")
pipeline.set_state(Gst.State.PLAYING)
time.sleep(1)

# Example 1: Load model with custom settings
print("\n=== Example 1: Load Model ===")
send_control_command(control_pad, {
    "command": "load_model",
    "model": "vibevoice/VibeVoice-1.5B",
    "compile": True,
    "diffusion_steps": 10
})

# Example 2: Load a voice sample
print("\n=== Example 2: Load Voice Sample ===")
send_control_command(control_pad, {
    "command": "load_voice",
    "name": "speaker1",
    "audio_path": "/path/to/voice/sample.wav"
})

# Example 3: Set the current voice
print("\n=== Example 3: Set Current Voice ===")
send_control_command(control_pad, {
    "command": "set_voice",
    "name": "speaker1"
})

# Example 4: Adjust generation parameters
print("\n=== Example 4: Adjust Parameters ===")
send_control_command(control_pad, {
    "command": "set_parameters",
    "cfg_scale": 2.0,
    "diffusion_steps": 15,
    "speed": 1.2
})

# Example 5: Query current status
print("\n=== Example 5: Get Status ===")
send_control_command(control_pad, {
    "command": "get_status"
})

# Example 6: Send text for generation
print("\n=== Example 6: Generate Speech ===")
sink_pad = vibevoice.get_static_pad("sink")
text = "This is a demonstration of runtime control using JSON commands."
buffer = Gst.Buffer.new_wrapped(text.encode('utf-8'))
sink_pad.chain(buffer)

# Let it play
time.sleep(3)

# Example 7: Interrupt generation (if it were still running)
print("\n=== Example 7: Interrupt Generation ===")
send_control_command(control_pad, {
    "command": "interrupt"
})

# Example 8: Change parameters mid-stream
print("\n=== Example 8: Change Parameters Again ===")
send_control_command(control_pad, {
    "command": "set_parameters",
    "cfg_scale": 1.3,
    "speed": 1.0
})

# Example 9: Load a different voice
print("\n=== Example 9: Load Another Voice ===")
send_control_command(control_pad, {
    "command": "load_voice",
    "name": "speaker2",
    "audio_path": "/path/to/another/sample.wav"
})
send_control_command(control_pad, {
    "command": "set_voice",
    "name": "speaker2"
})

# Example 10: Unload a voice
print("\n=== Example 10: Unload Voice ===")
send_control_command(control_pad, {
    "command": "unload_voice",
    "name": "speaker1"
})

# Example 11: Final status check
print("\n=== Example 11: Final Status ===")
send_control_command(control_pad, {
    "command": "get_status"
})

# Example 12: Unload model (cleanup)
print("\n=== Example 12: Unload Model ===")
send_control_command(control_pad, {
    "command": "unload_model"
})

# Send EOS
eos_event = Gst.Event.new_eos()
sink_pad.send_event(eos_event)

# Brief pause then cleanup
time.sleep(2)

# Cleanup
pipeline.set_state(Gst.State.NULL)
print("\nDone!")

# Control Pad Command Reference
print("\n" + "="*60)
print("CONTROL PAD COMMAND REFERENCE")
print("="*60)
print("""
1. load_model
   {"command": "load_model", "model": "model_name", "compile": true, "diffusion_steps": 10}

2. unload_model
   {"command": "unload_model"}

3. load_voice
   {"command": "load_voice", "name": "voice_name", "audio_path": "/path/to/sample.wav"}

4. unload_voice
   {"command": "unload_voice", "name": "voice_name"}

5. set_voice
   {"command": "set_voice", "name": "voice_name"}

6. set_parameters
   {"command": "set_parameters", "cfg_scale": 1.3, "diffusion_steps": 10, "speed": 1.0}

7. get_status
   {"command": "get_status"}

8. interrupt
   {"command": "interrupt"}
""")
