# VibeVoice Streaming TTS Assessment for GStreamer Integration

**Date**: 2025-11-07
**Target Use Case**: Real-time streaming TTS for GStreamer pipeline integration

---

## Executive Summary

**Verdict**: ⚠️ **CHALLENGING BUT POSSIBLE** with significant modifications

VibeVoice has streaming infrastructure in place, but it's designed for **long-form content generation** (podcasts, 90-minute audio), not low-latency real-time TTS. Key challenges:

- ❌ **High first-chunk latency** (requires full sentence processing before audio output)
- ❌ **Non-incremental text processing** (needs complete text upfront)
- ⚠️ **Moderate per-chunk latency** (~267ms per audio chunk + diffusion overhead)
- ✅ **Good streaming architecture** (queue-based, supports async)
- ✅ **Reasonable chunk sizes** (6400 samples/267ms @ 24kHz)

---

## Architecture Overview

### Current Pipeline Flow

```
Text Input (complete sentence/paragraph)
    ↓
Text Tokenization (Qwen2 tokenizer)
    ↓
Autoregressive Generation Loop:
    ├─ LM forward pass → generate special token
    ├─ If speech_diffusion token:
    │   ├─ Run diffusion (20 steps default, ~200-400ms)
    │   ├─ Generate 1 acoustic latent frame [1, 64]
    │   ├─ Decode to audio chunk (~6400 samples, 267ms @ 24kHz)
    │   ├─ Stream chunk to AudioStreamer queue
    │   └─ Encode to semantic features (feedback loop)
    └─ Repeat until speech_end or EOS
```

### Key Specs

| Metric | Value | Notes |
|--------|-------|-------|
| **Sample Rate** | 24,000 Hz | Fixed |
| **Hop Length** | 6400 samples | Product of ratios [8,5,5,4,2,2] |
| **Frame Duration** | ~267ms | 6400/24000 |
| **Frame Rate** | ~3.75 Hz | 24000/6400 |
| **Diffusion Steps** | 20 (configurable: 10-20) | Per audio chunk |
| **Chunk Size** | 6400 samples | ~267ms audio |
| **Model Sizes** | 1.5B (64K ctx) / 7B (32K ctx) | |

---

## Streaming Infrastructure Analysis

### ✅ What's Already Implemented

#### 1. AudioStreamer Class (`vibevoice/modular/streamer.py`)

```python
class AudioStreamer(BaseStreamer):
    - Per-sample Queue-based streaming
    - Supports batch inference (multiple samples)
    - Non-blocking iteration
    - Configurable timeouts
    - Clean stop signaling
```

**Key Features**:
- Audio chunks are queued immediately after generation (line 655 in inference)
- Supports both sync (`Queue`) and async (`asyncio.Queue`) modes
- Iterator interface for consuming audio stream
- Thread-safe queue operations

#### 2. Generation Loop Integration

The `generate()` method in `modeling_vibevoice_inference.py:327` includes:
- `audio_streamer` parameter support
- Immediate chunk dispatch: `audio_streamer.put(audio_chunk, diffusion_indices)` (line 655)
- Proper stream termination on EOS/max_length/stop signal

#### 3. Gradio Demo Streaming Example

`demo/gradio_demo.py:345-424` shows practical streaming:
- Buffers chunks (30 seconds worth) before first yield
- Streams subsequent chunks every 15 seconds OR 30 seconds of audio
- Real-time audio playback while generating

---

## ❌ Critical Limitations for Real-Time TTS

### 1. **No Incremental Text Input**

**Problem**: The model requires the **complete text** before starting generation.

```python
# Current API (inference_from_file.py)
inputs = processor(
    text=["Speaker 1: Complete sentence here"],  # ← Needs full text
    voice_samples=[[voice_path]],
    return_tensors="pt"
)
```

**Impact**:
- Cannot stream text tokens incrementally
- First chunk latency = text_processing + prefill + first_diffusion
- For sentence-level input, this could be **1-3 seconds**

**GStreamer Implication**: You'd need to buffer sentences/chunks before feeding to VibeVoice

---

### 2. **Autoregressive Token Generation**

The model generates tokens one-at-a-time sequentially:

```
Step 1: Generate token (maybe speech_start)
Step 2: Generate token (maybe speech_diffusion) → Run diffusion → Output audio chunk
Step 3: Generate token (maybe speech_diffusion) → Run diffusion → Output audio chunk
...
Step N: Generate token (speech_end)
```

**Latency Breakdown per Chunk**:
1. LM forward pass: ~50-100ms (1.5B) / ~150-300ms (7B)
2. Diffusion (20 steps): ~200-400ms (GPU dependent)
3. Acoustic decoding: ~10-50ms
4. **Total: ~260-450ms per 267ms audio chunk**

**Real-Time Factor (RTF)**: ~1.0-1.7x (NOT suitable for hard real-time)

---

### 3. **Voice Cloning Prefill Overhead**

If using voice cloning (`is_prefill=True`):
- Reference audio must be encoded (acoustic tokenizer encoder)
- Embeddings injected into initial input
- Adds **~100-300ms** startup latency

Can be disabled with `--disable_prefill`, but loses voice customization.

---

### 4. **CFG (Classifier-Free Guidance) Doubles Compute**

When `cfg_scale > 1.0` (default 1.3):
- Runs **two parallel forward passes** (positive + negative prompt)
- Effectively doubles LM inference time
- Can set to 1.0 to disable, but may reduce quality

---

## ⚠️ Latency Analysis

### First Chunk Latency (Cold Start)

```
Sentence input: "Speaker 1: Hello, how are you today?"

1. Text tokenization: ~10ms
2. Voice prefill (optional): ~100-300ms
3. LM prefill (process input): ~50-200ms (length dependent)
4. First token generation: ~50-100ms (1.5B) / ~150-300ms (7B)
5. If first audio token → diffusion: ~200-400ms
6. Acoustic decode: ~10-50ms

Estimated First Chunk: 420-1060ms (1.5B) / 570-1460ms (7B)
```

**For comparison**, traditional streaming TTS (e.g., Tacotron2 + WaveGlow):
- First chunk: 50-150ms

---

### Subsequent Chunks

After first chunk, each additional chunk:
- Token generation: ~50-100ms (1.5B) / ~150-300ms (7B)
- Diffusion: ~200-400ms
- Decode: ~10-50ms
- **Total: 260-450ms per 267ms audio**

**RTF**: 0.97-1.69 (7B model struggles to keep real-time)

---

## GStreamer Integration Strategy

### Architecture Proposal

```
┌─────────────────────────────────────────────────────────────┐
│                    GStreamer Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Text Source]                                              │
│       ↓                                                      │
│  [Sentence Splitter]  ← Custom element (split on .!?)       │
│       ↓                                                      │
│  [VibeVoice TTS Filter] ← Python/C++ hybrid                 │
│       │                                                      │
│       ├─ Python: VibeVoice inference with AudioStreamer     │
│       ├─ C wrapper: PyObject calls + audio queue reader     │
│       └─ GStreamer: Push audio buffers downstream           │
│       ↓                                                      │
│  [Audio Queue Buffer] ← Handle latency compensation         │
│       ↓                                                      │
│  [Audio Sink] (alsasink, pulsesink, etc.)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Recommendations

### Option 1: **C Extension with Python Embedding** (Recommended)

**Approach**:
- Write GStreamer element in C
- Embed Python interpreter (CPython API)
- Call VibeVoice Python API from C
- Read from AudioStreamer queue in C

**Pros**:
- Native GStreamer integration
- Good performance
- Control over threading and buffering

**Cons**:
- Complex (GIL management, memory management)
- Python embedding overhead

**Example Structure**:
```c
// gst-vibevoice.c
static GstFlowReturn gst_vibevoice_transform(GstBaseTransform *trans,
                                              GstBuffer *inbuf,
                                              GstBuffer *outbuf) {
    // 1. Read text from inbuf
    // 2. Call Python: model.generate(text, audio_streamer=streamer)
    // 3. Poll streamer queue from C
    // 4. Push audio chunks to outbuf
    return GST_FLOW_OK;
}
```

---

### Option 2: **Python GStreamer Element** (Easier)

**Approach**:
- Use `python-gstreamer` bindings
- Write element entirely in Python
- Direct VibeVoice API access

**Pros**:
- Much simpler development
- Full Python ecosystem access
- Rapid prototyping

**Cons**:
- Performance overhead (Python in streaming pipeline)
- GIL contention
- Less control over memory/threads

**Example**:
```python
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

class VibeVoiceTTS(Gst.Element):
    __gstmetadata__ = (
        "VibeVoice TTS",
        "Filter/Audio",
        "Text-to-speech using VibeVoice",
        "Your Name"
    )

    def do_transform(self, inbuf, outbuf):
        text = inbuf.extract_dup(0, inbuf.get_size()).decode('utf-8')

        # Generate audio
        streamer = AudioStreamer(batch_size=1)
        outputs = self.model.generate(
            input_ids=self.processor(text=text),
            audio_streamer=streamer
        )

        # Push chunks
        for chunk in streamer.get_stream(0):
            audio_data = chunk.numpy().tobytes()
            buf = Gst.Buffer.new_allocate(None, len(audio_data), None)
            buf.fill(0, audio_data)
            self.srcpad.push(buf)

        return Gst.FlowReturn.OK
```

---

### Option 3: **Subprocess with IPC**

**Approach**:
- GStreamer element spawns Python subprocess
- Communicate via pipes/shared memory
- VibeVoice runs in separate process

**Pros**:
- Clean separation (no GIL issues)
- Can crash/restart Python without killing pipeline
- Better for long-running services

**Cons**:
- IPC latency overhead
- More complex architecture
- Need robust error handling

---

## Code Modifications Required

### 1. **Add Sentence-Level Streaming API**

Currently, you must provide full text. Need to add:

```python
# New API in modeling_vibevoice_inference.py
def generate_streaming_text(
    self,
    text_iterator: Iterator[str],  # Stream sentences
    audio_streamer: AudioStreamer,
    **kwargs
) -> Iterator[torch.Tensor]:
    """
    Accept text chunks incrementally and stream audio.
    """
    for text_chunk in text_iterator:
        # Process chunk
        inputs = self.processor(text=[text_chunk])
        # Generate and stream
        yield from self.generate(..., audio_streamer=audio_streamer)
```

**Challenge**: The model's autoregressive nature means you can't truly "stream" text tokens mid-sentence without restarting generation.

---

### 2. **Optimize Diffusion Steps**

Current default: 20 steps. Can reduce to 10:

```python
model.set_ddpm_inference_steps(num_steps=10)
```

**Impact**:
- ~50% faster audio generation
- Slight quality degradation
- RTF improves from 1.0-1.7x to 0.5-0.85x

---

### 3. **Disable CFG for Speed**

```python
outputs = model.generate(..., cfg_scale=1.0)  # Disable CFG
```

**Impact**:
- ~50% faster (single forward pass)
- May reduce expressiveness

---

### 4. **Batch Pipelining** (Advanced)

Process multiple sentences concurrently:

```
Thread 1: Sentence A → Audio chunks 1-5
Thread 2: Sentence B → Audio chunks 6-10 (starts while A is finishing)
Thread 3: Sentence C → ...
```

Requires careful queue management and GPU memory.

---

## Performance Optimizations

### Hardware Recommendations

| Model | Minimum | Recommended |
|-------|---------|-------------|
| **1.5B** | RTX 3060 (12GB) | RTX 4070 (12GB) |
| **7B** | RTX 4080 (16GB) | RTX 4090 (24GB) |

**For real-time (RTF < 1.0)**:
- Use 1.5B model
- 10 diffusion steps
- CFG disabled (cfg_scale=1.0)
- Flash Attention 2 enabled
- BF16 precision

---

### Quantization Options

Not officially supported, but could explore:
- INT8 quantization (via `bitsandbytes`)
- Would reduce quality but improve speed ~2x

---

## Alternative Approach: Hybrid TTS

**Idea**: Use VibeVoice for high-quality segments, fallback to fast TTS for real-time

```
GStreamer Pipeline:
    ↓
[Priority Router]
    ├─ High-priority text → FastTTS (Piper, Coqui) [50ms latency]
    └─ Low-priority text → VibeVoice [1s latency]
```

Use VibeVoice for:
- Pre-generated content
- Non-interactive audio
- Background narration

Use fast TTS for:
- Interactive responses
- Real-time dialogue
- Low-latency requirements

---

## Proof of Concept: Minimal GStreamer Element

### Python GStreamer Element (Start Here)

```python
#!/usr/bin/env python3
"""
Minimal VibeVoice GStreamer element
Usage: gst-launch-1.0 filesrc location=text.txt ! vibevoicetts ! audioconvert ! autoaudiosink
"""

import gi
import torch
import numpy as np
from queue import Queue, Empty
from threading import Thread

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase

from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer

Gst.init(None)

class VibeVoiceTTS(GstBase.BaseTransform):
    __gstmetadata__ = (
        "VibeVoice TTS Filter",
        "Filter/Audio/TTS",
        "Convert text to speech using VibeVoice",
        "Your Name <your@email.com>"
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("text/x-raw, format=utf8")
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string("audio/x-raw, format=F32LE, rate=24000, channels=1")
        )
    )

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None
        self.initialized = False

    def do_start(self):
        """Initialize model (called when pipeline starts)"""
        if not self.initialized:
            print("Loading VibeVoice model...")
            self.processor = VibeVoiceProcessor.from_pretrained("vibevoice/VibeVoice-1.5B")
            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                "vibevoice/VibeVoice-1.5B",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=10)  # Fast mode
            self.initialized = True
            print("Model loaded!")
        return True

    def do_transform(self, inbuf, outbuf):
        """Transform text buffer to audio buffer"""
        # Extract text
        text_data = inbuf.extract_dup(0, inbuf.get_size())
        text = text_data.decode('utf-8').strip()

        if not text:
            return Gst.FlowReturn.OK

        print(f"Generating audio for: {text[:50]}...")

        # Setup streamer
        streamer = AudioStreamer(batch_size=1)

        # Generate audio in background thread
        def generate():
            inputs = self.processor(
                text=[f"Speaker 1: {text}"],
                voice_samples=[[None]],  # No voice cloning
                return_tensors="pt"
            ).to(self.model.device)

            self.model.generate(
                **inputs,
                audio_streamer=streamer,
                cfg_scale=1.0,  # Disable CFG for speed
                tokenizer=self.processor.tokenizer,
                is_prefill=False,
                show_progress_bar=False
            )

        gen_thread = Thread(target=generate, daemon=True)
        gen_thread.start()

        # Collect all audio chunks
        audio_chunks = []
        for chunk in streamer.get_stream(0):
            if torch.is_tensor(chunk):
                audio_np = chunk.cpu().float().numpy()
            else:
                audio_np = np.array(chunk, dtype=np.float32)
            audio_chunks.append(audio_np.flatten())

        gen_thread.join()

        # Concatenate and write to output buffer
        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            audio_bytes = full_audio.tobytes()
            outbuf.fill(0, audio_bytes)
            outbuf.resize(len(audio_bytes))
            print(f"Generated {len(full_audio)/24000:.2f}s of audio")

        return Gst.FlowReturn.OK

# Register element
GObject.type_register(VibeVoiceTTS)
__gstelementfactory__ = ("vibevoicetts", Gst.Rank.NONE, VibeVoiceTTS)
```

**Save as**: `gst_vibevoice_plugin.py`

**Test**:
```bash
# Set plugin path
export GST_PLUGIN_PATH=$PWD

# Test pipeline
echo "Hello world, this is a test." | \
gst-launch-1.0 fdsrc ! \
    vibevoicetts ! \
    audioconvert ! \
    audioresample ! \
    autoaudiosink
```

---

## C Wrapper Skeleton

If you want to go the C route:

```c
/* gst-vibevoice.c */
#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <Python.h>

typedef struct _GstVibeVoice {
    GstBaseTransform parent;

    /* Python objects */
    PyObject *py_model;
    PyObject *py_processor;
    PyObject *py_streamer_class;
} GstVibeVoice;

static GstFlowReturn
gst_vibevoice_transform(GstBaseTransform *trans, GstBuffer *inbuf, GstBuffer *outbuf)
{
    GstVibeVoice *self = GST_VIBEVOICE(trans);
    GstMapInfo map;
    gst_buffer_map(inbuf, &map, GST_MAP_READ);

    /* Extract text */
    gchar *text = g_strndup((gchar *)map.data, map.size);
    gst_buffer_unmap(inbuf, &map);

    /* Call Python model */
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject *py_text = PyUnicode_FromString(text);
    PyObject *py_streamer = PyObject_CallFunctionObjArgs(
        self->py_streamer_class,
        PyLong_FromLong(1),  /* batch_size=1 */
        NULL
    );

    /* Call model.generate(...) */
    PyObject *result = PyObject_CallMethod(
        self->py_model,
        "generate",
        "OO",  /* text, audio_streamer */
        py_text,
        py_streamer
    );

    /* Read from streamer queue */
    PyObject *stream = PyObject_CallMethod(py_streamer, "get_stream", "i", 0);
    PyObject *iterator = PyObject_GetIter(stream);

    GstBuffer *audio_buf = gst_buffer_new();

    while (TRUE) {
        PyObject *chunk = PyIter_Next(iterator);
        if (!chunk) break;

        /* Convert numpy/tensor to bytes */
        /* TODO: Handle numpy array / torch tensor conversion */
        /* Append to audio_buf */
    }

    Py_DECREF(iterator);
    Py_DECREF(stream);
    PyGILState_Release(gstate);

    gst_buffer_copy_into(outbuf, audio_buf, GST_BUFFER_COPY_ALL, 0, -1);
    g_free(text);

    return GST_FLOW_OK;
}

/* Standard GStreamer plugin boilerplate... */
```

---

## Testing & Benchmarking

### Latency Measurement

Add instrumentation to measure:

```python
import time

# In generate() method
times = {
    'start': time.time(),
    'first_token': None,
    'first_audio': None,
    'chunks': []
}

# Before first LM forward
times['prefill_start'] = time.time()

# After first token
if times['first_token'] is None:
    times['first_token'] = time.time()
    print(f"First token: {times['first_token'] - times['start']:.3f}s")

# In streaming loop
if audio_streamer:
    chunk_time = time.time()
    audio_streamer.put(audio_chunk, diffusion_indices)
    times['chunks'].append(chunk_time - times['start'])
    if times['first_audio'] is None:
        times['first_audio'] = chunk_time
        print(f"First audio chunk: {times['first_audio'] - times['start']:.3f}s")
```

### Target Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| First chunk latency | < 1000ms | < 500ms |
| RTF (Real-Time Factor) | < 1.0 | < 0.5 |
| Chunk latency | < 300ms | < 150ms |
| Memory usage | < 8GB VRAM | < 6GB |

---

## Conclusion & Recommendations

### ✅ **Feasible for Your Use Case IF:**

1. **Latency tolerance**: You can accept 0.5-1.5s first-chunk delay
2. **Sentence buffering**: You buffer complete sentences before feeding to model
3. **GPU available**: You have RTX 3060+ or better
4. **Optimization willingness**: You tune (10 diffusion steps, CFG off, etc.)

### ❌ **NOT Recommended IF:**

1. **Hard real-time needed**: You need <200ms response time
2. **CPU-only**: Model is too slow without GPU
3. **Interactive dialogue**: User expects immediate TTS response
4. **Token-by-token streaming**: You want to stream partial words

### 🎯 **Best Approach**:

**Start with Option 2 (Python GStreamer Element)**:
1. Prototype quickly in Python
2. Measure actual latencies on your hardware
3. Optimize (reduce diffusion steps, disable CFG, etc.)
4. If performance is acceptable → Done!
5. If not → Migrate to C wrapper (Option 1) or consider hybrid approach

**Write C wrappers** only if:
- Python version works but needs 20-30% more performance
- You need tighter control over memory/threading
- You're building a production system

---

## Next Steps

1. **Benchmark on your hardware**: Run inference_from_file.py and measure RTF
2. **Test streaming**: Run gradio_demo.py and observe chunk latencies
3. **Prototype Python element**: Use skeleton code above
4. **Optimize**: Tune diffusion steps, CFG, model size
5. **Evaluate**: Does it meet your latency requirements?
6. **Decide**: Python element sufficient? Or need C wrapper?

---

## Contact & Resources

- **Discord**: https://discord.gg/ZDEYTTRxWG
- **GitHub**: https://github.com/vibevoice-community/VibeVoice
- **GStreamer Python**: https://gstreamer.freedesktop.org/documentation/application-development/advanced/pipeline-manipulation.html

**Good luck with your integration!** Feel free to reach out to the community if you hit roadblocks.
