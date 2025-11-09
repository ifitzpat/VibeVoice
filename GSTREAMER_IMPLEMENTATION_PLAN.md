# GStreamer VibeVoice Element - Implementation Plan

**Date**: 2025-11-09
**Target**: Production-quality GStreamer element for VibeVoice TTS

---

## Executive Summary

**Goal**: Create `gst-vibevoice`, a GStreamer element that:
- Accepts text sentences on sink pad
- Outputs 24kHz audio to src pad (webrtc-sink/pipewire compatible)
- Provides control pad for runtime configuration
- Supports model/voice loading, parameter tuning
- Includes internal buffering for smooth operation

**Language**: Python (for rapid development, easy VibeVoice integration)

**Future**: Will be spun off to separate repository once working

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GstVibeVoice Element                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐      ┌─────────────────┐      ┌──────────┐      │
│  │  Sink    │─────→│  Sentence Queue │─────→│   Src    │      │
│  │  Pad     │      │   (buffering)   │      │   Pad    │      │
│  │ (text)   │      └─────────┬───────┘      │ (audio)  │      │
│  └──────────┘                │              └──────────┘      │
│                               │                                 │
│  ┌──────────┐                ↓                                 │
│  │ Control  │      ┌─────────────────┐                         │
│  │  Pad     │─────→│  Model Manager  │                         │
│  │(commands)│      │  - Load/unload  │                         │
│  └──────────┘      │  - Voice mgmt   │                         │
│                    │  - Parameters   │                         │
│                    └────────┬────────┘                         │
│                             │                                   │
│                             ↓                                   │
│                    ┌─────────────────┐                         │
│                    │  VibeVoice      │                         │
│                    │  Inference      │                         │
│                    │  (with compile) │                         │
│                    └─────────────────┘                         │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Background Thread (TTS Generation)                      │  │
│  │  - Consumes from sentence queue                          │  │
│  │  - Generates audio chunks                                │  │
│  │  - Pushes to src pad                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pad Design

### 1. Sink Pad (Text Input)

**Capabilities**:
```python
Gst.Caps.from_string(
    "text/x-raw, format=(string)utf8; "
    "application/x-subtitle; "
    "text/plain"
)
```

**Accepts**:
- UTF-8 text sentences
- One sentence per buffer (recommended)
- Or continuous text stream (will be sentence-split internally)

**Behavior**:
- Non-blocking push to internal queue
- Returns `GST_FLOW_OK` immediately
- Actual TTS happens in background thread

---

### 2. Src Pad (Audio Output)

**Capabilities**:
```python
Gst.Caps.from_string(
    "audio/x-raw, "
    "format=(string)F32LE, "  # 32-bit float, little-endian
    "rate=(int)24000, "
    "channels=(int)1, "
    "layout=(string)interleaved"
)
```

**Output Format**:
- Sample rate: 24,000 Hz (VibeVoice native)
- Format: 32-bit float (F32LE) or 16-bit signed int (S16LE)
- Channels: 1 (mono)
- Chunk size: Variable (based on VibeVoice output)

**Timestamps**:
- Proper PTS (Presentation Timestamp) for each buffer
- Duration set based on audio length
- Continuous stream (no gaps)

---

### 3. Control Pad (Runtime Configuration)

**Capabilities**:
```python
Gst.Caps.from_string(
    "application/x-vibevoice-control"
)
```

**Message Format**: JSON for flexibility

**Supported Commands**:

```json
{
  "command": "load_model",
  "model": "vibevoice/VibeVoice-1.5B",
  "compile": true,
  "diffusion_steps": 10
}

{
  "command": "unload_model"
}

{
  "command": "load_voice",
  "name": "Alice",
  "audio_path": "/path/to/alice_reference.wav"
}

{
  "command": "unload_voice",
  "name": "Alice"
}

{
  "command": "set_voice",
  "name": "Alice"
}

{
  "command": "set_parameters",
  "cfg_scale": 1.3,
  "diffusion_steps": 10,
  "speed": 1.0
}

{
  "command": "get_status"
}
```

**Response**: Element posts messages to bus
```python
# Status message
message = Gst.Message.new_application(
    element,
    Gst.Structure.new(
        "vibevoice-status",
        ("model_loaded", "s", "VibeVoice-1.5B"),
        ("current_voice", "s", "Alice"),
        ("vram_usage_gb", "d", 7.2),
        ("queue_length", "i", 3)
    )
)
self.post_message(message)
```

---

## Internal Architecture

### State Management

```python
class GstVibeVoice(Gst.Element):
    """Main element class"""

    def __init__(self):
        super().__init__()

        # Pads
        self.sinkpad = None
        self.srcpad = None
        self.controlpad = None

        # Model state
        self.model = None
        self.processor = None
        self.model_name = None
        self.compiled = False

        # Voice management
        self.voices = {}  # name -> audio_path mapping
        self.current_voice = None

        # Parameters
        self.cfg_scale = 1.3
        self.diffusion_steps = 10
        self.speed = 1.0

        # Buffering
        self.sentence_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=50)

        # Threading
        self.generator_thread = None
        self.pusher_thread = None
        self.running = False

        # Synchronization
        self.lock = threading.Lock()

        # Timing
        self.base_time = 0
        self.current_timestamp = 0
        self.sample_count = 0
```

---

### Threading Model

**Three threads**:

1. **Main GStreamer Thread** (receives buffers, handles controls)
2. **Generator Thread** (runs VibeVoice inference)
3. **Pusher Thread** (pushes audio to src pad)

**Flow**:
```
Main Thread                Generator Thread           Pusher Thread
     │                            │                         │
     │ Text buffer received       │                         │
     │──────────────────────────→ │                         │
     │   (push to sentence_queue) │                         │
     │                            │                         │
     │                            │ Pop sentence            │
     │                            │ Generate audio (TTS)    │
     │                            │ Push to audio_queue ───→│
     │                            │                         │
     │                            │                         │ Pop audio chunk
     │                            │                         │ Push to src pad
     │                            │                         │ Update timestamp
     │                            │                         │
     │ Control message            │                         │
     │──────────────────────────→ │                         │
     │   (update parameters)      │                         │
```

**Why three threads?**
- **Main thread**: Must return quickly (GStreamer requirement)
- **Generator thread**: VibeVoice inference is slow (200-500ms)
- **Pusher thread**: Smooth audio output with proper timing

---

### Buffering Strategy

**Sentence Queue** (input buffering):
```python
self.sentence_queue = queue.Queue(maxsize=10)
```
- Max 10 sentences queued
- Blocks if full (backpressure)
- FIFO order

**Audio Queue** (output buffering):
```python
self.audio_queue = queue.Queue(maxsize=50)
```
- Stores generated audio chunks
- Max 50 chunks (~13 seconds @ 267ms/chunk)
- Smooths output timing

**Watermark Management**:
```python
LOW_WATERMARK = 5   # Start generating when queue < 5 chunks
HIGH_WATERMARK = 40  # Pause generating when queue > 40 chunks
```

---

## File Structure

```
vibevoice/gstreamer/
├── __init__.py
├── plugin.py                    # Plugin registration
├── element.py                   # Main GstVibeVoice element
├── model_manager.py             # Model loading/unloading logic
├── voice_manager.py             # Voice reference management
├── control_handler.py           # Control pad message handling
├── audio_generator.py           # Background TTS generation thread
├── audio_pusher.py              # Audio output thread
├── utils.py                     # Utilities (timestamp calc, etc.)
└── tests/
    ├── test_element.py
    ├── test_control.py
    └── test_pipeline.py

# Later: Separate repository
gst-vibevoice/
├── src/
│   └── gst_vibevoice/
│       ├── __init__.py
│       ├── element.py
│       ├── model_manager.py
│       └── ...
├── tests/
├── examples/
│   ├── basic_pipeline.py
│   ├── webrtc_example.py
│   └── control_example.py
├── setup.py
├── README.md
└── LICENSE
```

---

## Detailed Component Design

### 1. Plugin Registration (`plugin.py`)

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

from .element import GstVibeVoice

def plugin_init(plugin):
    """Plugin initialization function"""
    type_to_register = GObject.type_register(GstVibeVoice)

    return Gst.Element.register(
        plugin,
        "vibevoice",           # Element name
        Gst.Rank.NONE,         # Rank (for autoplugging)
        type_to_register
    )

# Plugin metadata
GST_PLUGIN_DEFINE = {
    "name": "vibevoice",
    "description": "VibeVoice Text-to-Speech element",
    "version": "0.1.0",
    "license": "MIT",
    "source": "gst-vibevoice",
    "package": "GStreamer VibeVoice Plugin",
    "origin": "https://github.com/yourusername/gst-vibevoice",
    "init": plugin_init
}
```

---

### 2. Main Element (`element.py`)

```python
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
import threading
import queue
import time

from .model_manager import ModelManager
from .voice_manager import VoiceManager
from .control_handler import ControlHandler
from .audio_generator import AudioGenerator
from .audio_pusher import AudioPusher

class GstVibeVoice(Gst.Element):
    """
    GStreamer element for VibeVoice TTS.

    Pads:
        sink: Text input (text/x-raw, utf8)
        src: Audio output (audio/x-raw, F32LE, 24000Hz, mono)
        control: Control messages (application/x-vibevoice-control)
    """

    __gstmetadata__ = (
        "VibeVoice TTS",
        "Filter/Audio/TTS",
        "Text-to-speech using VibeVoice",
        "Your Name <your@email.com>"
    )

    # Pad templates
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "text/x-raw, format=(string)utf8; "
                "text/plain"
            )
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "audio/x-raw, "
                "format=(string)F32LE, "
                "rate=(int)24000, "
                "channels=(int)1, "
                "layout=(string)interleaved"
            )
        ),
        Gst.PadTemplate.new(
            "control",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string(
                "application/x-vibevoice-control"
            )
        )
    )

    # Properties (GObject properties for gst-inspect)
    __gproperties__ = {
        "model": (
            str,
            "Model name",
            "VibeVoice model to load",
            "vibevoice/VibeVoice-1.5B",
            GObject.ParamFlags.READWRITE
        ),
        "voice": (
            str,
            "Current voice",
            "Active voice for synthesis",
            None,
            GObject.ParamFlags.READWRITE
        ),
        "cfg-scale": (
            float,
            "CFG scale",
            "Classifier-free guidance scale",
            0.0, 5.0, 1.3,
            GObject.ParamFlags.READWRITE
        ),
        "diffusion-steps": (
            int,
            "Diffusion steps",
            "Number of diffusion steps (10-20)",
            5, 50, 10,
            GObject.ParamFlags.READWRITE
        ),
        "compile": (
            bool,
            "Use torch.compile",
            "Apply torch.compile optimization",
            True,
            GObject.ParamFlags.READWRITE
        ),
        "queue-size": (
            int,
            "Sentence queue size",
            "Maximum sentences to buffer",
            1, 100, 10,
            GObject.ParamFlags.READWRITE
        )
    }

    def __init__(self):
        super().__init__()

        # Create pads
        self.sinkpad = Gst.Pad.new_from_template(
            self.__gsttemplates__[0], "sink"
        )
        self.sinkpad.set_chain_function_full(self.chain_text)
        self.sinkpad.set_event_function_full(self.sink_event)
        self.add_pad(self.sinkpad)

        self.srcpad = Gst.Pad.new_from_template(
            self.__gsttemplates__[1], "src"
        )
        self.srcpad.set_event_function_full(self.src_event)
        self.add_pad(self.srcpad)

        # Control pad created on request
        self.controlpad = None

        # Managers
        self.model_manager = ModelManager()
        self.voice_manager = VoiceManager()
        self.control_handler = ControlHandler(self)

        # Queues
        self.sentence_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=50)

        # Threads
        self.generator = None
        self.pusher = None
        self.running = False

        # State
        self.lock = threading.Lock()
        self.eos_received = False

        # Timing
        self.sample_rate = 24000
        self.current_timestamp = 0
        self.base_time = None

        # Properties
        self._model_name = "vibevoice/VibeVoice-1.5B"
        self._voice_name = None
        self._cfg_scale = 1.3
        self._diffusion_steps = 10
        self._use_compile = True
        self._queue_size = 10

        Gst.info(f"VibeVoice element created")

    def do_request_new_pad(self, template, name, caps):
        """Handle control pad creation"""
        if template.name_template == "control":
            if self.controlpad is not None:
                Gst.warning("Control pad already exists")
                return None

            self.controlpad = Gst.Pad.new_from_template(template, "control")
            self.controlpad.set_chain_function_full(self.chain_control)
            self.add_pad(self.controlpad)

            Gst.info("Control pad created")
            return self.controlpad

        return None

    def do_change_state(self, transition):
        """Handle state changes"""
        if transition == Gst.StateChange.NULL_TO_READY:
            # Initialize model (optional: can defer to first buffer)
            pass

        elif transition == Gst.StateChange.READY_TO_PAUSED:
            # Start threads
            self.start_threads()

        elif transition == Gst.StateChange.PAUSED_TO_PLAYING:
            # Resume if needed
            pass

        # Call parent
        result = Gst.Element.do_change_state(self, transition)

        if transition == Gst.StateChange.PLAYING_TO_PAUSED:
            # Pause if needed
            pass

        elif transition == Gst.StateChange.PAUSED_TO_READY:
            # Stop threads
            self.stop_threads()

        elif transition == Gst.StateChange.READY_TO_NULL:
            # Cleanup
            self.cleanup()

        return result

    def start_threads(self):
        """Start background threads"""
        with self.lock:
            if self.running:
                return

            self.running = True
            self.eos_received = False

            # Generator thread
            self.generator = AudioGenerator(
                self.sentence_queue,
                self.audio_queue,
                self.model_manager,
                self.voice_manager,
                self
            )
            self.generator.start()

            # Pusher thread
            self.pusher = AudioPusher(
                self.audio_queue,
                self.srcpad,
                self
            )
            self.pusher.start()

            Gst.info("Background threads started")

    def stop_threads(self):
        """Stop background threads"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # Signal threads to stop
            self.sentence_queue.put(None)  # Poison pill

            # Wait for completion
            if self.generator:
                self.generator.join(timeout=5.0)
            if self.pusher:
                self.pusher.join(timeout=5.0)

            Gst.info("Background threads stopped")

    def cleanup(self):
        """Cleanup resources"""
        self.model_manager.unload_model()
        self.voice_manager.clear()

        # Clear queues
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def chain_text(self, pad, parent, buffer):
        """Handle incoming text buffer"""
        # Extract text
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            Gst.error("Failed to map buffer")
            return Gst.FlowReturn.ERROR

        try:
            text = map_info.data.decode('utf-8').strip()
        finally:
            buffer.unmap(map_info)

        if not text:
            return Gst.FlowReturn.OK

        # Queue sentence for processing
        try:
            self.sentence_queue.put(text, block=True, timeout=1.0)
            Gst.debug(f"Queued text: {text[:50]}...")
        except queue.Full:
            Gst.warning("Sentence queue full, dropping buffer")
            return Gst.FlowReturn.OK  # Or ERROR to signal backpressure

        return Gst.FlowReturn.OK

    def chain_control(self, pad, parent, buffer):
        """Handle control messages"""
        # Extract JSON command
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            command_json = map_info.data.decode('utf-8')
        finally:
            buffer.unmap(map_info)

        # Process command
        self.control_handler.handle_command(command_json)

        return Gst.FlowReturn.OK

    def sink_event(self, pad, parent, event):
        """Handle sink pad events"""
        if event.type == Gst.EventType.EOS:
            Gst.info("EOS received on sink pad")
            self.eos_received = True
            # Signal generator thread
            self.sentence_queue.put(None)
            return True

        elif event.type == Gst.EventType.CAPS:
            caps = event.parse_caps()
            Gst.info(f"Caps event: {caps.to_string()}")
            return True

        elif event.type == Gst.EventType.SEGMENT:
            segment = event.parse_segment()
            self.base_time = segment.time
            return True

        # Forward other events
        return self.srcpad.push_event(event)

    def src_event(self, pad, parent, event):
        """Handle src pad events"""
        # Forward upstream
        return self.sinkpad.push_event(event)

    # GObject property handlers
    def do_get_property(self, prop):
        if prop.name == "model":
            return self._model_name
        elif prop.name == "voice":
            return self._voice_name
        elif prop.name == "cfg-scale":
            return self._cfg_scale
        elif prop.name == "diffusion-steps":
            return self._diffusion_steps
        elif prop.name == "compile":
            return self._use_compile
        elif prop.name == "queue-size":
            return self._queue_size
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "model":
            self._model_name = value
        elif prop.name == "voice":
            self._voice_name = value
        elif prop.name == "cfg-scale":
            self._cfg_scale = value
        elif prop.name == "diffusion-steps":
            self._diffusion_steps = value
        elif prop.name == "compile":
            self._use_compile = value
        elif prop.name == "queue-size":
            self._queue_size = value
            # Resize queue if needed
            # (Note: Python Queue doesn't support resizing, would need custom implementation)
        else:
            raise AttributeError(f"Unknown property {prop.name}")
```

---

### 3. Model Manager (`model_manager.py`)

```python
import torch
import gc
from typing import Optional

class ModelManager:
    """Manages VibeVoice model loading, unloading, and compilation"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None
        self.compiled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str, compile: bool = True, diffusion_steps: int = 10):
        """Load VibeVoice model"""
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        # Unload existing model
        if self.model is not None:
            self.unload_model()

        print(f"Loading model: {model_name}")

        # Load processor
        self.processor = VibeVoiceProcessor.from_pretrained(model_name)

        # Load model
        self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager"
        )
        self.model.eval()
        self.model.set_ddpm_inference_steps(diffusion_steps)

        self.model_name = model_name

        # Compile if requested
        if compile and self.device == "cuda":
            self.compile_model()

        print(f"Model loaded: {model_name}")

    def compile_model(self):
        """Apply torch.compile optimization"""
        if self.compiled:
            return

        print("Compiling model with torch.compile...")

        # Compile key components
        self.model.model.acoustic_tokenizer.decoder = torch.compile(
            self.model.model.acoustic_tokenizer.decoder,
            mode="reduce-overhead",
            fullgraph=False
        )

        self.model.model.prediction_head = torch.compile(
            self.model.model.prediction_head,
            mode="reduce-overhead",
            fullgraph=False
        )

        self.model.model.decoder = torch.compile(
            self.model.model.decoder,
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True
        )

        # Warmup
        print("Warming up compiled model...")
        dummy_inputs = self.processor(
            text=["Warmup"],
            voice_samples=[[None]],
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            _ = self.model.generate(
                **dummy_inputs,
                tokenizer=self.processor.tokenizer,
                max_new_tokens=5,
                is_prefill=False,
                show_progress_bar=False
            )

        self.compiled = True
        print("Model compilation complete")

    def unload_model(self):
        """Unload model and free VRAM"""
        if self.model is None:
            return

        print("Unloading model...")

        # Move to CPU
        if self.device == "cuda":
            self.model.to("cpu")

        # Delete
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self.model_name = None
        self.compiled = False

        # Cleanup
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model unloaded")

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def generate(self, text: str, voice_sample: Optional[str] = None,
                 cfg_scale: float = 1.3, **kwargs):
        """Generate audio for text"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        # Prepare inputs
        inputs = self.processor(
            text=[text],
            voice_samples=[[voice_sample]] if voice_sample else [[None]],
            return_tensors="pt"
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                cfg_scale=cfg_scale,
                is_prefill=(voice_sample is not None),
                show_progress_bar=False,
                **kwargs
            )

        # Extract audio
        if outputs.speech_outputs and len(outputs.speech_outputs[0]) > 0:
            audio = outputs.speech_outputs[0][0].cpu().float().numpy()
            return audio
        else:
            return None
```

---

### 4. Voice Manager (`voice_manager.py`)

```python
import os
from typing import Dict, Optional

class VoiceManager:
    """Manages voice reference audio files"""

    def __init__(self):
        self.voices: Dict[str, str] = {}  # name -> audio_path
        self.current_voice = None

    def load_voice(self, name: str, audio_path: str):
        """Load a voice reference"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Voice audio not found: {audio_path}")

        self.voices[name] = audio_path
        print(f"Voice '{name}' loaded from {audio_path}")

    def unload_voice(self, name: str):
        """Unload a voice reference"""
        if name in self.voices:
            del self.voices[name]
            print(f"Voice '{name}' unloaded")

            if self.current_voice == name:
                self.current_voice = None

    def set_current_voice(self, name: str):
        """Set the active voice"""
        if name not in self.voices:
            raise ValueError(f"Voice '{name}' not loaded")

        self.current_voice = name
        print(f"Current voice set to '{name}'")

    def get_current_voice_path(self) -> Optional[str]:
        """Get the current voice audio path"""
        if self.current_voice is None:
            return None
        return self.voices.get(self.current_voice)

    def list_voices(self):
        """List all loaded voices"""
        return list(self.voices.keys())

    def clear(self):
        """Clear all voices"""
        self.voices.clear()
        self.current_voice = None
```

---

### 5. Control Handler (`control_handler.py`)

```python
import json
from gi.repository import Gst

class ControlHandler:
    """Handles control pad messages"""

    def __init__(self, element):
        self.element = element

    def handle_command(self, command_json: str):
        """Process a control command"""
        try:
            command = json.loads(command_json)
        except json.JSONDecodeError as e:
            Gst.error(f"Invalid JSON in control message: {e}")
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
        else:
            Gst.warning(f"Unknown command: {cmd_type}")
            self.send_error(f"Unknown command: {cmd_type}")

    def handle_load_model(self, command):
        """Load model command"""
        model_name = command.get("model", "vibevoice/VibeVoice-1.5B")
        compile = command.get("compile", True)
        diffusion_steps = command.get("diffusion_steps", 10)

        try:
            self.element.model_manager.load_model(
                model_name,
                compile=compile,
                diffusion_steps=diffusion_steps
            )
            self.send_status(f"Model loaded: {model_name}")
        except Exception as e:
            Gst.error(f"Failed to load model: {e}")
            self.send_error(f"Model load failed: {e}")

    def handle_unload_model(self):
        """Unload model command"""
        try:
            self.element.model_manager.unload_model()
            self.send_status("Model unloaded")
        except Exception as e:
            Gst.error(f"Failed to unload model: {e}")
            self.send_error(f"Model unload failed: {e}")

    def handle_load_voice(self, command):
        """Load voice command"""
        name = command.get("name")
        audio_path = command.get("audio_path")

        if not name or not audio_path:
            self.send_error("Missing 'name' or 'audio_path' in load_voice command")
            return

        try:
            self.element.voice_manager.load_voice(name, audio_path)
            self.send_status(f"Voice loaded: {name}")
        except Exception as e:
            Gst.error(f"Failed to load voice: {e}")
            self.send_error(f"Voice load failed: {e}")

    def handle_unload_voice(self, command):
        """Unload voice command"""
        name = command.get("name")

        if not name:
            self.send_error("Missing 'name' in unload_voice command")
            return

        try:
            self.element.voice_manager.unload_voice(name)
            self.send_status(f"Voice unloaded: {name}")
        except Exception as e:
            Gst.error(f"Failed to unload voice: {e}")
            self.send_error(f"Voice unload failed: {e}")

    def handle_set_voice(self, command):
        """Set current voice command"""
        name = command.get("name")

        if not name:
            self.send_error("Missing 'name' in set_voice command")
            return

        try:
            self.element.voice_manager.set_current_voice(name)
            self.send_status(f"Current voice set to: {name}")
        except Exception as e:
            Gst.error(f"Failed to set voice: {e}")
            self.send_error(f"Set voice failed: {e}")

    def handle_set_parameters(self, command):
        """Set parameters command"""
        if "cfg_scale" in command:
            self.element._cfg_scale = command["cfg_scale"]

        if "diffusion_steps" in command:
            self.element._diffusion_steps = command["diffusion_steps"]
            if self.element.model_manager.is_loaded():
                self.element.model_manager.model.set_ddpm_inference_steps(
                    command["diffusion_steps"]
                )

        if "speed" in command:
            self.element._speed = command["speed"]

        self.send_status("Parameters updated")

    def handle_get_status(self):
        """Get status command"""
        import torch

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

        if torch.cuda.is_available():
            status["vram_usage_gb"] = torch.cuda.memory_allocated() / 1e9

        self.send_status(json.dumps(status, indent=2))

    def send_status(self, message: str):
        """Send status message to bus"""
        struct = Gst.Structure.new_empty("vibevoice-status")
        struct.set_value("message", message)

        msg = Gst.Message.new_application(self.element, struct)
        self.element.post_message(msg)

    def send_error(self, message: str):
        """Send error message to bus"""
        struct = Gst.Structure.new_empty("vibevoice-error")
        struct.set_value("message", message)

        msg = Gst.Message.new_application(self.element, struct)
        self.element.post_message(msg)
```

---

## Implementation Phases

### Phase 1: Basic Element (Week 1)
- ✅ Element registration and pad setup
- ✅ Basic text input → audio output
- ✅ Model manager (load/unload)
- ✅ Simple synchronous generation (no threading yet)
- ✅ Manual testing with gst-launch

**Deliverable**: Working element that can do basic TTS

---

### Phase 2: Threading & Buffering (Week 2)
- ✅ Implement sentence queue
- ✅ Implement audio queue
- ✅ Add generator thread
- ✅ Add pusher thread
- ✅ Proper timestamp management
- ✅ Test with longer inputs

**Deliverable**: Smooth streaming audio output

---

### Phase 3: Control Pad (Week 3)
- ✅ Control pad implementation
- ✅ JSON command protocol
- ✅ Voice manager
- ✅ Control handler
- ✅ Status messages on bus
- ✅ Test all control commands

**Deliverable**: Fully controllable element

---

### Phase 4: torch.compile Integration (Week 4)
- ✅ Implement torch.compile in model_manager
- ✅ Add compilation during model load
- ✅ Proper warmup strategy (background thread to avoid blocking)
- ✅ Benchmark before/after compilation
- ✅ Add `compile` property to element
- ✅ Handle compilation failures gracefully
- ✅ Cache compiled models across reloads (optional)

**Deliverable**: 20-40% faster inference with torch.compile

---

### Phase 5: VRAM & Resource Management (Week 5)
- ✅ Dynamic model loading/unloading via control pad
- ✅ VRAM usage monitoring and reporting
- ✅ Proper cleanup on state changes
- ✅ Memory leak testing (24+ hour stress test)
- ✅ CPU fallback when CUDA unavailable
- ✅ Graceful degradation on errors

**Deliverable**: Robust resource management

---

### Phase 6: Integration & Documentation (Week 6)
- ✅ WebRTC example
- ✅ Pipewire example
- ✅ Control examples
- ✅ API documentation
- ✅ README with usage
- ✅ Unit tests

**Deliverable**: Ready for separate repository

---

## Testing Strategy

### Unit Tests

```python
# tests/test_element.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pytest

def test_element_creation():
    """Test element can be created"""
    Gst.init(None)
    element = Gst.ElementFactory.make("vibevoice", "tts")
    assert element is not None

def test_pad_templates():
    """Test pad templates are correct"""
    element = Gst.ElementFactory.make("vibevoice", "tts")

    # Check sink pad
    sinkpad = element.get_static_pad("sink")
    assert sinkpad is not None

    # Check src pad
    srcpad = element.get_static_pad("src")
    assert srcpad is not None

def test_basic_generation():
    """Test basic text-to-audio generation"""
    # Create pipeline
    pipeline = Gst.parse_launch(
        "fakesrc num-buffers=1 ! "
        "text/x-raw,format=utf8 ! "
        "vibevoice name=tts ! "
        "fakesink"
    )

    # Run
    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.SECOND * 10,
        Gst.MessageType.EOS | Gst.MessageType.ERROR
    )

    assert msg.type == Gst.MessageType.EOS
```

---

### Integration Tests

```python
# tests/test_pipeline.py
def test_webrtc_pipeline():
    """Test integration with webrtcbin"""
    pipeline = Gst.parse_launch(
        "filesrc location=test.txt ! "
        "vibevoice ! "
        "audioconvert ! "
        "webrtcbin name=sendonly ! "
        "fakesink"
    )
    # ...

def test_pipewire_pipeline():
    """Test integration with pipewire"""
    pipeline = Gst.parse_launch(
        "filesrc location=test.txt ! "
        "vibevoice ! "
        "audioconvert ! "
        "audio/x-raw,format=F32LE,rate=48000 ! "
        "pipewiresink"
    )
    # ...
```

---

## Usage Examples

### Basic Pipeline

```bash
# Simple file-based TTS
gst-launch-1.0 \
  filesrc location=input.txt ! \
  vibevoice model="vibevoice/VibeVoice-1.5B" compile=true ! \
  audioconvert ! \
  autoaudiosink
```

---

### WebRTC Pipeline

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

# Create pipeline
pipeline = Gst.Pipeline()

# Text source (from your application)
appsrc = Gst.ElementFactory.make("appsrc", "src")
appsrc.set_property("format", Gst.Format.TIME)
appsrc.set_property("is-live", True)

# VibeVoice TTS
tts = Gst.ElementFactory.make("vibevoice", "tts")
tts.set_property("model", "vibevoice/VibeVoice-1.5B")
tts.set_property("compile", True)

# Audio conversion for WebRTC
convert = Gst.ElementFactory.make("audioconvert", "convert")
resample = Gst.ElementFactory.make("audioresample", "resample")
capsfilter = Gst.ElementFactory.make("capsfilter", "caps")
capsfilter.set_property("caps", Gst.Caps.from_string(
    "audio/x-raw,format=S16LE,rate=48000,channels=1"
))

# WebRTC
webrtc = Gst.ElementFactory.make("webrtcbin", "webrtc")

# Add to pipeline
pipeline.add(appsrc, tts, convert, resample, capsfilter, webrtc)

# Link
appsrc.link(tts)
tts.link(convert)
convert.link(resample)
resample.link(capsfilter)
capsfilter.link(webrtc)

# Send text
def send_text(text):
    buffer = Gst.Buffer.new_wrapped(text.encode('utf-8'))
    appsrc.emit("push-buffer", buffer)

# Start
pipeline.set_state(Gst.State.PLAYING)
send_text("Hello from WebRTC!")
```

---

### Control Pad Example

```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import json

Gst.init(None)

# Create pipeline
pipeline = Gst.Pipeline()
tts = Gst.ElementFactory.make("vibevoice", "tts")
sink = Gst.ElementFactory.make("autoaudiosink", "sink")

pipeline.add(tts, sink)
tts.link(sink)

# Request control pad
control_pad = tts.request_pad_simple("control")

# Send control commands
def send_control(command_dict):
    command_json = json.dumps(command_dict)
    buffer = Gst.Buffer.new_wrapped(command_json.encode('utf-8'))
    control_pad.chain(buffer)

# Load model
send_control({
    "command": "load_model",
    "model": "vibevoice/VibeVoice-1.5B",
    "compile": True,
    "diffusion_steps": 10
})

# Load voices
send_control({
    "command": "load_voice",
    "name": "Alice",
    "audio_path": "/path/to/alice.wav"
})

send_control({
    "command": "load_voice",
    "name": "Frank",
    "audio_path": "/path/to/frank.wav"
})

# Set current voice
send_control({
    "command": "set_voice",
    "name": "Alice"
})

# Listen for status messages
bus = pipeline.get_bus()
def on_message(bus, message):
    if message.type == Gst.MessageType.APPLICATION:
        struct = message.get_structure()
        if struct.has_name("vibevoice-status"):
            print(f"Status: {struct.get_value('message')}")
        elif struct.has_name("vibevoice-error"):
            print(f"Error: {struct.get_value('message')}")
    return True

bus.add_watch(GLib.PRIORITY_DEFAULT, on_message)

# Start pipeline
pipeline.set_state(Gst.State.PLAYING)

# Send text
text_pad = tts.get_static_pad("sink")
text_buffer = Gst.Buffer.new_wrapped(b"Hello, this is Alice speaking.")
text_pad.chain(text_buffer)

# Switch voice mid-stream
send_control({
    "command": "set_voice",
    "name": "Frank"
})

text_buffer2 = Gst.Buffer.new_wrapped(b"And now this is Frank.")
text_pad.chain(text_buffer2)

# Get status
send_control({"command": "get_status"})

# Run main loop
loop = GLib.MainLoop()
loop.run()
```

---

## Next Steps

1. **Create file structure** in `vibevoice/gstreamer/`
2. **Implement Phase 1** (basic element)
3. **Test with gst-launch-1.0**
4. **Iterate on feedback**
5. **Continue through phases**
6. **Spin off to separate repo** when ready

---

## Questions to Resolve

1. **Audio format**: F32LE or S16LE for output? (F32LE is VibeVoice native, S16LE is more compatible)
2. **Queue sizes**: 10 sentences / 50 chunks good defaults?
3. **Error handling**: Drop buffers on queue full, or block and signal backpressure?
4. **Voice loading**: Allow loading from file paths only, or support in-memory audio?
5. **Property vs Control pad**: Some settings (like cfg_scale) can be both - which to prioritize?

Let me know your preferences and I can start implementing!
