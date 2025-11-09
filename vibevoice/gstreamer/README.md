# VibeVoice GStreamer Plugin

A production-quality GStreamer plugin for the VibeVoice text-to-speech model. This plugin enables real-time TTS generation with support for voice cloning, runtime configuration, and optimized inference.

## Features

- **GStreamer Integration**: Standard GStreamer element with sink (text input) and src (audio output) pads
- **Runtime Control**: JSON-based control pad for dynamic configuration
- **Voice Cloning**: Load and manage multiple voice samples
- **Optimized Inference**: torch.compile support for 20-40% faster generation on CUDA
- **Threaded Architecture**: Non-blocking audio generation with buffered output
- **Resource Management**: Dynamic model loading/unloading with VRAM monitoring
- **Streaming Output**: Real-time audio streaming as text is generated

## Installation

### Prerequisites

```bash
# GStreamer
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good python3-gi gstreamer1.0-python3-plugin-loader

# PyTorch with CUDA (for GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# VibeVoice (install from repository root)
cd /path/to/VibeVoice
pip install -e .
```

### Plugin Installation

The plugin is automatically available when you install the VibeVoice package. GStreamer will find it through the Python plugin loader.

Verify installation:

```bash
gst-inspect-1.0 vibevoice
```

## Quick Start

### Basic Usage

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

Gst.init(None)

# Create pipeline
pipeline = Gst.parse_launch(
    "vibevoice name=tts ! audioconvert ! autoaudiosink"
)

# Configure element
vibevoice = pipeline.get_by_name("tts")
vibevoice.set_property("model-name", "vibevoice/VibeVoice-1.5B")
vibevoice.set_property("compile", True)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Send text
sinkpad = vibevoice.get_static_pad("sink")
text = "Hello! This is VibeVoice speaking."
buffer = Gst.Buffer.new_wrapped(text.encode('utf-8'))
sinkpad.chain(buffer)

# Signal end of input
eos_event = Gst.Event.new_eos()
sinkpad.send_event(eos_event)

# Run main loop
loop = GLib.MainLoop()
loop.run()
```

### Command Line Usage

```bash
# Basic text-to-speech
echo "Hello world" | gst-launch-1.0 \
    fdsrc ! vibevoice model-name=vibevoice/VibeVoice-1.5B ! \
    audioconvert ! autoaudiosink

# Save to file
echo "Hello world" | gst-launch-1.0 \
    fdsrc ! vibevoice ! audioconvert ! \
    wavenc ! filesink location=output.wav
```

## Element Reference

### Pads

#### Sink Pad (text input)
- **Name**: `sink`
- **Direction**: Sink
- **Capabilities**: `text/plain`
- **Description**: Accepts UTF-8 encoded text for speech generation

#### Source Pad (audio output)
- **Name**: `src`
- **Direction**: Source
- **Capabilities**: `audio/x-raw, format=F32LE, rate=24000, channels=1`
- **Description**: Outputs 32-bit float audio at 24kHz mono

#### Control Pad (JSON commands)
- **Name**: `control`
- **Direction**: Sink
- **Capabilities**: `text/plain`
- **Description**: Accepts JSON-formatted control commands

### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model-name` | String | `vibevoice/VibeVoice-1.5B` | HuggingFace model name or path |
| `compile` | Boolean | `True` | Enable torch.compile optimization (CUDA only) |
| `cfg-scale` | Float | `1.3` | Classifier-free guidance scale (higher = more faithful to text) |
| `diffusion-steps` | Integer | `10` | Number of diffusion steps (higher = better quality, slower) |
| `speed` | Float | `1.0` | Playback speed multiplier |

### Signals

#### ready
```python
def on_ready(element, is_ready):
    """Emitted when model loading completes or element becomes ready/unready"""
    pass

vibevoice.connect("ready", on_ready)
```

#### model-loading
```python
def on_model_loading(element, model_name):
    """Emitted when model loading starts"""
    pass

vibevoice.connect("model-loading", on_model_loading)
```

#### model-loaded
```python
def on_model_loaded(element, model_name):
    """Emitted when model loading completes successfully"""
    pass

vibevoice.connect("model-loaded", on_model_loaded)
```

#### model-unloaded
```python
def on_model_unloaded(element):
    """Emitted when model is unloaded and VRAM is freed"""
    pass

vibevoice.connect("model-unloaded", on_model_unloaded)
```

#### compilation-started
```python
def on_compilation_started(element):
    """Emitted when torch.compile optimization begins"""
    pass

vibevoice.connect("compilation-started", on_compilation_started)
```

#### compilation-finished
```python
def on_compilation_finished(element):
    """Emitted when torch.compile optimization completes"""
    pass

vibevoice.connect("compilation-finished", on_compilation_finished)
```

#### voice-loaded
```python
def on_voice_loaded(element, voice_name):
    """Emitted when a voice sample is loaded"""
    pass

vibevoice.connect("voice-loaded", on_voice_loaded)
```

#### voice-changed
```python
def on_voice_changed(element, voice_name):
    """Emitted when the current voice is changed"""
    pass

vibevoice.connect("voice-changed", on_voice_changed)
```

#### generation-started
```python
def on_generation_started(element, text):
    """Emitted when audio generation starts for a sentence"""
    pass

vibevoice.connect("generation-started", on_generation_started)
```

#### generation-complete
```python
def on_generation_complete(element):
    """Emitted when all queued generation is complete"""
    pass

vibevoice.connect("generation-complete", on_generation_complete)
```

#### generation-error
```python
def on_generation_error(element, error_message):
    """Emitted when an error occurs during generation"""
    pass

vibevoice.connect("generation-error", on_generation_error)
```

## Control Pad Protocol

The control pad accepts JSON commands for runtime configuration. This allows dynamic adjustment of parameters, voice switching, and model management without restarting the pipeline.

### Commands

#### load_model
Load a VibeVoice model from HuggingFace or local path.

```json
{
  "command": "load_model",
  "model": "vibevoice/VibeVoice-1.5B",
  "compile": true,
  "diffusion_steps": 10
}
```

**Parameters:**
- `model` (string, optional): Model name or path. Default: `vibevoice/VibeVoice-1.5B`
- `compile` (boolean, optional): Enable torch.compile. Default: `true`
- `diffusion_steps` (integer, optional): Number of diffusion steps. Default: `10`

#### unload_model
Unload the current model and free VRAM.

```json
{
  "command": "unload_model"
}
```

#### load_voice
Load a voice sample for voice cloning.

```json
{
  "command": "load_voice",
  "name": "speaker1",
  "audio_path": "/path/to/voice/sample.wav"
}
```

**Parameters:**
- `name` (string, required): Unique identifier for this voice
- `audio_path` (string, required): Path to audio file (WAV, MP3, etc.)

#### unload_voice
Unload a previously loaded voice sample.

```json
{
  "command": "unload_voice",
  "name": "speaker1"
}
```

**Parameters:**
- `name` (string, required): Name of voice to unload

#### set_voice
Set the active voice for generation.

```json
{
  "command": "set_voice",
  "name": "speaker1"
}
```

**Parameters:**
- `name` (string, required): Name of loaded voice to use

#### set_parameters
Adjust generation parameters at runtime.

```json
{
  "command": "set_parameters",
  "cfg_scale": 1.3,
  "diffusion_steps": 10,
  "speed": 1.0
}
```

**Parameters:**
- `cfg_scale` (float, optional): Classifier-free guidance scale
- `diffusion_steps` (integer, optional): Number of diffusion steps
- `speed` (float, optional): Playback speed multiplier

#### get_status
Query the current status of the element.

```json
{
  "command": "get_status"
}
```

**Response** (via `vibevoice-status` bus message):
```json
{
  "model_loaded": true,
  "model_name": "vibevoice/VibeVoice-1.5B",
  "compiled": true,
  "current_voice": "speaker1",
  "loaded_voices": ["speaker1", "speaker2"],
  "cfg_scale": 1.3,
  "diffusion_steps": 10,
  "sentence_queue_size": 0,
  "audio_queue_size": 0,
  "vram_usage_gb": 3.2
}
```

#### interrupt
Interrupt current generation and clear all queues.

```json
{
  "command": "interrupt"
}
```

### Bus Messages

The element posts application messages to the GStreamer bus:

- **vibevoice-status**: Status updates and responses to commands
- **vibevoice-error**: Error messages

Example message handling:

```python
def on_message(bus, message):
    if message.type == Gst.MessageType.APPLICATION:
        struct = message.get_structure()
        if struct.get_name() == "vibevoice-status":
            status = struct.get_value("message")
            print(f"Status: {status}")
        elif struct.get_name() == "vibevoice-error":
            error = struct.get_value("message")
            print(f"Error: {error}")
    return True

bus = pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message)
```

## Advanced Usage

### Voice Cloning

```python
import json

# Get control pad
control_pad = vibevoice.get_static_pad("control")

# Load voice sample
command = json.dumps({
    "command": "load_voice",
    "name": "my_voice",
    "audio_path": "/path/to/voice/sample.wav"
})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)

# Set as active voice
command = json.dumps({
    "command": "set_voice",
    "name": "my_voice"
})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)

# Now all generated speech will use this voice
```

### Dynamic Parameter Adjustment

```python
import json

control_pad = vibevoice.get_static_pad("control")

# Increase quality mid-stream
command = json.dumps({
    "command": "set_parameters",
    "diffusion_steps": 20,
    "cfg_scale": 1.5
})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)
```

### VRAM Management

```python
import json

control_pad = vibevoice.get_static_pad("control")

# Check VRAM usage
command = json.dumps({"command": "get_status"})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)

# Unload model to free VRAM
command = json.dumps({"command": "unload_model"})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)

# Reload when needed
command = json.dumps({"command": "load_model"})
buffer = Gst.Buffer.new_wrapped(command.encode('utf-8'))
control_pad.chain(buffer)
```

## Performance Optimization

### torch.compile

Enable torch.compile for 20-40% faster inference on CUDA:

```python
vibevoice.set_property("compile", True)
```

Note: First inference will be slower due to compilation. Subsequent calls are faster.

### Diffusion Steps

Trade off quality vs. speed by adjusting diffusion steps:

```python
# Faster, lower quality
vibevoice.set_property("diffusion-steps", 5)

# Balanced (default)
vibevoice.set_property("diffusion-steps", 10)

# Slower, higher quality
vibevoice.set_property("diffusion-steps", 20)
```

### Batching

The element automatically batches sentences from the input text for efficient generation.

## Architecture

### Threading Model

```
Input Text → Sentence Queue → Generator Thread → Audio Queue → Pusher Thread → Output
                                      ↓
                                 Model Manager
                                 Voice Manager
```

- **Generator Thread**: Processes sentences, generates audio using VibeVoice model
- **Pusher Thread**: Pushes audio buffers to src pad with proper timestamps
- **Interrupt Flag**: Allows graceful interruption of generation

### Resource Management

- Model loading/unloading on demand
- Automatic VRAM cleanup with `torch.cuda.empty_cache()`
- Graceful degradation to CPU if CUDA unavailable
- Thread-safe queue management

## Examples

See the `examples/` directory for complete examples:

- **basic_usage.py**: Simple text-to-speech pipeline
- **control_example.py**: Comprehensive control pad demonstration

Run examples:

```bash
python3 vibevoice/gstreamer/examples/basic_usage.py
python3 vibevoice/gstreamer/examples/control_example.py
```

## Testing

The plugin includes comprehensive unit tests:

```bash
# Run all tests
python3 -m pytest vibevoice/gstreamer/tests/ -v

# Run specific test file
python3 -m pytest vibevoice/gstreamer/tests/test_element.py -v

# Run with coverage
python3 -m pytest vibevoice/gstreamer/tests/ --cov=vibevoice.gstreamer --cov-report=html
```

## Troubleshooting

### Plugin Not Found

If `gst-inspect-1.0 vibevoice` fails:

```bash
# Check Python plugin loader is installed
gst-inspect-1.0 python

# Verify VibeVoice package is installed
python3 -c "import vibevoice.gstreamer.element"

# Check GStreamer plugin path
export GST_PLUGIN_PATH=$GST_PLUGIN_PATH:~/.local/lib/python3.12/site-packages
```

### CUDA Out of Memory

```python
# Reduce model size or unload when not in use
control_pad.chain(Gst.Buffer.new_wrapped(
    json.dumps({"command": "unload_model"}).encode('utf-8')
))

# Disable compilation to save VRAM
vibevoice.set_property("compile", False)
```

### Slow First Inference

When torch.compile is enabled, the first inference is slow due to compilation. This is expected. Subsequent inferences are 20-40% faster.

### No Audio Output

Check pipeline state:

```python
# Ensure pipeline is PLAYING
state = pipeline.get_state(Gst.CLOCK_TIME_NONE)
print(f"Pipeline state: {state}")

# Check for errors on bus
bus = pipeline.get_bus()
message = bus.timed_pop_filtered(
    Gst.CLOCK_TIME_NONE,
    Gst.MessageType.ERROR | Gst.MessageType.EOS
)
if message:
    print(f"Message: {message.type}")
```

## License

See the main VibeVoice repository for license information.

## Contributing

Contributions are welcome! Please see the main repository for contribution guidelines.

## Support

For issues, questions, or feature requests, please use the main VibeVoice repository issue tracker.
