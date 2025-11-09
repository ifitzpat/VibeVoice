# VibeVoice Mobile Deployment & VRAM Management Strategy

**Date**: 2025-11-09
**Context**: RTX 5090 (32GB) running VibeVoice + LLM + Whisper.cpp with future iOS/Android deployment

---

## Executive Summary

**Your Constraints**:
- ✅ RTX 5090 with 32GB VRAM
- ⚠️ Running: VibeVoice + Local LLM + Whisper simultaneously
- 🎯 Need dynamic model loading/unloading
- 📱 Future target: iOS/Android deployment

**Recommended Strategy**: **Dual-Model Approach**

| Platform | TTS Engine | Reason |
|----------|------------|--------|
| **Desktop/Server** | VibeVoice (Python) | Exceptional quality, your 5090 handles it well |
| **Mobile (iOS/Android)** | Piper via Sherpa-ONNX | ONNX-native, C++ first-class, proven mobile support |

**Key Insight**: Don't force VibeVoice onto mobile. Use the right tool for each platform.

---

## VRAM Budget Analysis (RTX 5090 - 32GB)

### Typical VRAM Usage

| Model | Size | Precision | VRAM (Approx) | Notes |
|-------|------|-----------|---------------|-------|
| **VibeVoice-1.5B** | 1.5B params | BF16 | 6-8 GB | With KV-cache + diffusion |
| **VibeVoice-7B** | 7B params | BF16 | 14-16 GB | With KV-cache + diffusion |
| **LLM (Llama-7B)** | 7B params | BF16 | 14-15 GB | With KV-cache (4K context) |
| **LLM (Llama-13B)** | 13B params | BF16 | 26-28 GB | **Won't fit with others** |
| **LLM (Mistral-7B)** | 7B params | BF16 | 14-15 GB | Similar to Llama |
| **Whisper Large-v3** | 1.5B params | FP16 | 3-4 GB | Running in whisper.cpp |
| **Whisper Medium** | 769M params | FP16 | 1.5-2 GB | More conservative |

### Realistic Scenarios

#### Scenario A: High-Quality Voice Agent (Comfortable)
```
VibeVoice-1.5B:  6-8 GB
Llama-7B:        14-15 GB
Whisper Large:   3-4 GB
---------------------------------
Total:           23-27 GB ✅ FITS COMFORTABLY (5-9GB free)
```

**Verdict**: Plenty of headroom! Can run all three models simultaneously.

#### Scenario B: Ultra-Quality Voice Agent (Recommended)
```
VibeVoice-7B:    14-16 GB
Llama-7B/13B:    14-26 GB
Whisper Large:   3-4 GB
---------------------------------
Total:           31-46 GB ⚠️ 7B fits, 13B needs dynamic loading
```

**Verdict**:
- VibeVoice-7B + Llama-7B + Whisper: ✅ Fits (~31GB, 1GB free)
- VibeVoice-7B + Llama-13B: ❌ Need dynamic loading

#### Scenario C: Maximum Quality (Dynamic Loading)
```
VibeVoice-7B:    14-16 GB
Llama-13B:       26-28 GB
(Load one at a time)
---------------------------------
Peak:            26-28 GB ✅ FITS with dynamic loading
```

**Solution**: Keep LLM resident, load VibeVoice on-demand for TTS.

#### Scenario D: Fast Response (Mobile Target)
```
Piper TTS:       <200 MB
Llama-7B (GGUF Q4): 4-5 GB
Whisper Medium:  1.5-2 GB
---------------------------------
Total:           6-7 GB ✅ PLENTY OF HEADROOM
```

**Verdict**: This is your mobile-like configuration on desktop for testing.

---

## Python Model Loading/Unloading Strategies

### Basic Pattern: Sequential Loading

```python
import gc
import torch
from transformers import AutoModelForCausalLM
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

class ModelManager:
    def __init__(self):
        self.current_model = None
        self.model_name = None

    def load_model(self, model_type):
        """Load a model and unload previous one"""
        # Unload existing model
        if self.current_model is not None:
            self.unload_current()

        # Load new model
        if model_type == "vibevoice":
            self.current_model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                "vibevoice/VibeVoice-1.5B",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
        elif model_type == "llm":
            self.current_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-7B",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )

        self.model_name = model_type
        return self.current_model

    def unload_current(self):
        """Properly unload model and free VRAM"""
        if self.current_model is None:
            return

        # Move model to CPU first (optional but recommended)
        self.current_model.to("cpu")

        # Delete model
        del self.current_model
        self.current_model = None
        self.model_name = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        torch.cuda.empty_cache()

        # Optional: Wait a bit for GPU to settle
        torch.cuda.synchronize()

        print(f"VRAM freed: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

# Usage
manager = ModelManager()

# Load VibeVoice for TTS
vv_model = manager.load_model("vibevoice")
# ... generate audio
manager.unload_current()

# Load LLM for chat
llm_model = manager.load_model("llm")
# ... generate response
manager.unload_current()
```

### Advanced: Overlapping Pipeline with Queues

For your GStreamer use case, you might want models to persist but swap priority:

```python
import threading
from queue import Queue
from contextlib import contextmanager

class PipelineManager:
    """
    Manages multiple models with dynamic priority.
    Keeps frequently-used models loaded, swaps out infrequent ones.
    """
    def __init__(self, vram_budget_gb=20):
        self.models = {}
        self.vram_usage = {}
        self.vram_budget = vram_budget_gb * 1e9
        self.lock = threading.Lock()

    def estimate_vram(self, model_type):
        """Estimate VRAM for a model"""
        estimates = {
            "vibevoice-1.5b": 7e9,
            "vibevoice-7b": 15e9,
            "llm-7b": 14e9,
            "whisper-large": 3.5e9,
            "whisper-medium": 2e9,
        }
        return estimates.get(model_type, 5e9)

    def get_current_vram(self):
        """Get actual VRAM usage"""
        return torch.cuda.memory_allocated()

    def free_space_if_needed(self, required_vram):
        """Unload least-recently-used models to free space"""
        current = self.get_current_vram()
        available = self.vram_budget - current

        if available >= required_vram:
            return True

        # Sort by last access time (implement LRU logic)
        # Unload models until we have space
        # (Implementation left as exercise)

    @contextmanager
    def use_model(self, model_type, model_key):
        """Context manager for using a model"""
        with self.lock:
            # Check if model is loaded
            if model_key not in self.models:
                required = self.estimate_vram(model_type)
                self.free_space_if_needed(required)

                # Load model
                if model_type == "vibevoice-1.5b":
                    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                        "vibevoice/VibeVoice-1.5B",
                        torch_dtype=torch.bfloat16,
                        device_map="cuda"
                    )
                # ... other model types

                self.models[model_key] = model
                self.vram_usage[model_key] = self.get_current_vram()

            model = self.models[model_key]

        try:
            yield model
        finally:
            # Update access time for LRU
            pass

# Usage
pipeline = PipelineManager(vram_budget_gb=22)

# In GStreamer TTS element
with pipeline.use_model("vibevoice-1.5b", "tts") as model:
    audio = model.generate(...)

# In LLM processing
with pipeline.use_model("llm-7b", "chat") as model:
    response = model.generate(...)
```

### Strategy: Keep Core Models Resident

For your pipeline, consider:

**Always Loaded** (resident):
- **Whisper** (1.5-2 GB) - needed for all ASR
- **LLM** (14-15 GB) - core conversation engine

**Load on Demand**:
- **VibeVoice** (6-8 GB) - only when generating TTS

**Total resident**: ~17-19 GB, leaving ~5-7 GB for VibeVoice when needed.

```python
class VoiceAgentPipeline:
    def __init__(self):
        # Always loaded
        self.whisper = load_whisper("medium")  # 2GB
        self.llm = load_llm("llama-7b")  # 14GB

        # Loaded on demand
        self.tts = None

    def transcribe(self, audio):
        return self.whisper.transcribe(audio)

    def chat(self, text):
        return self.llm.generate(text)

    def speak(self, text):
        # Load TTS if not present
        if self.tts is None:
            self.tts = VibeVoiceForConditionalGenerationInference.from_pretrained(
                "vibevoice/VibeVoice-1.5B",
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )

        audio = self.tts.generate(text, ...)
        return audio

    def unload_tts(self):
        """Call after TTS to free VRAM for other tasks"""
        if self.tts is not None:
            del self.tts
            self.tts = None
            gc.collect()
            torch.cuda.empty_cache()
```

---

## Mobile Deployment: iOS/Android

### Reality Check: Python on Mobile

**iOS**:
- ❌ No official Python runtime in App Store apps
- ❌ App Store restrictions on dynamic code execution
- ⚠️ Workarounds exist (BeeWare, Kivy) but clunky and unreliable
- ❌ VibeVoice requires PyTorch (huge binary size)

**Android**:
- ⚠️ Python via Kivy/Chaquopy/Termux (hacky)
- ❌ Large APK size (PyTorch ~100MB+)
- ❌ Battery/thermal constraints
- ❌ Not ideal for production apps

**Verdict**: **Don't deploy Python/PyTorch models to mobile production apps.**

---

## Mobile TTS Solution: Piper via Sherpa-ONNX

### Why Sherpa-ONNX?

**Sherpa-ONNX** (https://github.com/k2-fsa/sherpa-onnx) is specifically designed for mobile:

✅ Supports **iOS, Android, HarmonyOS**
✅ Uses **ONNX Runtime** (C++ native, small binary)
✅ Loads **Piper models** (ONNX format)
✅ **Offline inference** (no internet required)
✅ Low latency (~50-150ms)
✅ Small model size (20-200MB)
✅ **12 programming languages** (C++, Swift, Kotlin, Java, Python, etc.)

### iOS Deployment with Sherpa-ONNX

```swift
// iOS Swift example
import SherpaOnnx

class TTSManager {
    var tts: SherpaOnnxOfflineTts?

    func initialize() {
        var config = sherpaOnnxOfflineTtsConfig()

        // Configure Piper model
        config.model.vits.model = Bundle.main.path(forResource: "en_US-lessac-medium", ofType: "onnx")!
        config.model.vits.tokens = Bundle.main.path(forResource: "tokens", ofType: "txt")!
        config.model.vits.dataDir = Bundle.main.bundlePath

        config.model.numThreads = 2
        config.model.provider = "cpu"  // or "coreml" for Apple Neural Engine

        tts = SherpaOnnxOfflineTts(config: config)
    }

    func speak(text: String) -> [Float] {
        guard let tts = tts else { return [] }

        let audio = tts.generate(text: text, sid: 0, speed: 1.0)
        return audio.samples
    }
}
```

**Binary Size**: ~10-15MB framework + 20-200MB model = **30-215MB total**

**Performance on iPhone 14 Pro**:
- Latency: 50-100ms first chunk
- RTF: 0.05-0.15 (very fast)
- Runs on CPU or ANE (Apple Neural Engine)

### Android Deployment with Sherpa-ONNX

```kotlin
// Android Kotlin example
import com.k2fsa.sherpa.onnx.*

class TTSManager(context: Context) {
    private var tts: OfflineTts? = null

    fun initialize() {
        val config = OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                vits = OfflineTtsVitsModelConfig(
                    model = "assets://en_US-lessac-medium.onnx",
                    tokens = "assets://tokens.txt",
                    dataDir = context.filesDir.absolutePath
                ),
                numThreads = 2,
                provider = "cpu"  // or "nnapi" for hardware acceleration
            )
        )

        tts = OfflineTts(config)
    }

    fun speak(text: String): FloatArray {
        return tts?.generate(text, sid = 0, speed = 1.0f)?.samples ?: floatArrayOf()
    }
}
```

**APK Size**: ~15-20MB library + 20-200MB model = **35-220MB total**

**Performance on Pixel 8**:
- Latency: 50-120ms first chunk
- RTF: 0.08-0.2
- Runs on CPU or NNAPI (Android Neural Networks API)

### Available Piper Models for Mobile

| Model | Size | Quality | Languages | Use Case |
|-------|------|---------|-----------|----------|
| `*-low.onnx` | 20-40MB | Basic | Most | Ultra-lightweight |
| `*-medium.onnx` | 60-100MB | Good | Most | **Recommended** |
| `*-high.onnx` | 150-200MB | Very Good | Some | High quality |

**60+ voices** in 30+ languages available via Piper.

---

## Universal vs. Dual-Model Strategy

### Option 1: Universal Solution (Port VibeVoice)

**Goal**: Single codebase, same model everywhere

**Reality**:
- ❌ 4-6 weeks to port to C++/ONNX
- ❌ Model size: ~3-6GB (too large for mobile)
- ❌ VRAM requirements: 6-8GB (mobile GPUs have 4-8GB shared with system)
- ❌ Battery drain on mobile
- ❌ Maintenance burden for C++ port
- ❌ Loses future VibeVoice Python updates

**Verdict**: **Not recommended** - VibeVoice is designed for server/desktop, not mobile.

---

### Option 2: Dual-Model Strategy (Recommended)

**Goal**: Right tool for each platform

#### Desktop/Server Stack

```python
# GStreamer pipeline on Linux/Windows with RTX 5090
VibeVoice-1.5B (Python)
  ├─ Quality: ⭐⭐⭐⭐⭐ (exceptional)
  ├─ Latency: 400-600ms first chunk
  ├─ VRAM: 6-8GB
  ├─ RTF: 0.3-0.4 on RTX 5090
  └─ Voice cloning: ✅ Multi-speaker
```

**GStreamer Integration**:
```python
# Python GStreamer element (as outlined in STREAMING_TTS_ASSESSMENT.md)
class VibeVoiceTTS(GstBase.BaseTransform):
    def do_transform(self, inbuf, outbuf):
        text = extract_text(inbuf)
        audio = self.vibevoice_model.generate(text, audio_streamer=streamer)
        push_audio_chunks(outbuf, audio)
```

#### Mobile Stack

```swift
// iOS with Sherpa-ONNX
Piper-Medium (ONNX)
  ├─ Quality: ⭐⭐⭐ (good)
  ├─ Latency: 50-100ms first chunk
  ├─ Binary: 80-120MB
  ├─ Runs on: CPU/Apple Neural Engine
  └─ Voice cloning: ❌ (but 60+ pre-trained voices)
```

```kotlin
// Android with Sherpa-ONNX
Piper-Medium (ONNX)
  ├─ Quality: ⭐⭐⭐ (good)
  ├─ Latency: 50-120ms first chunk
  ├─ APK: 95-220MB
  ├─ Runs on: CPU/NNAPI
  └─ Voice cloning: ❌
```

### Sharing Infrastructure Between Platforms

**Common Components**:
- Text preprocessing
- Sentence splitting
- SSML parsing (if you use it)
- Audio post-processing

**Platform-Specific**:
- TTS engine (VibeVoice vs Piper)
- Model loading
- VRAM management

**Example Shared Architecture**:

```
┌────────────────────────────────────────────────┐
│           Your Application Logic               │
│  (Text processing, conversation management)    │
└────────────────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Desktop/Server │         │  Mobile (iOS/   │
│   TTS Adapter   │         │     Android)    │
├─────────────────┤         ├─────────────────┤
│ VibeVoice-1.5B  │         │ Sherpa-ONNX +   │
│    (Python)     │         │  Piper (ONNX)   │
└─────────────────┘         └─────────────────┘
```

**Interface Design** (platform-agnostic):

```python
# Python (desktop)
class TTSAdapter:
    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Returns WAV audio bytes"""
        pass

# Desktop implementation
class VibeVoiceAdapter(TTSAdapter):
    def synthesize(self, text, voice_id):
        audio = self.model.generate(text, speaker=voice_id)
        return audio_to_wav(audio)

# Mobile would implement same interface with Piper
```

---

## Practical Implementation Plan

### Phase 1: Desktop Optimization (Now)

**Goal**: Maximize your RTX 5090 for desktop voice agent

1. **Keep these models resident**:
   ```python
   resident_models = {
       "whisper": "medium",  # 2GB
       "llm": "llama-7b",    # 14GB
   }
   # Total: ~16GB, leaving 8GB free
   ```

2. **Load VibeVoice on-demand**:
   ```python
   def handle_tts_request(text):
       # Load VibeVoice
       tts = load_vibevoice()  # 6-8GB
       audio = tts.generate(text)
       unload_vibevoice()      # Free 6-8GB
       return audio
   ```

3. **GStreamer Integration**:
   - Use Python GStreamer element (STREAMING_TTS_ASSESSMENT.md)
   - Implement model manager for loading/unloading
   - Add VRAM monitoring

**Expected Performance**:
- Whisper transcription: <100ms
- LLM response: 500-1500ms
- VibeVoice TTS: 400-600ms first chunk
- **Total latency**: ~1-2.5s (acceptable for voice agent)

### Phase 2: Mobile Prototyping (3-6 months)

**Goal**: Prove mobile TTS works well

1. **iOS Proof of Concept**:
   ```bash
   # Install Sherpa-ONNX iOS framework
   pod install SherpaOnnx

   # Download Piper model
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
   ```

2. **Android Proof of Concept**:
   ```bash
   # Add Sherpa-ONNX to build.gradle
   implementation 'com.k2fsa.sherpa:onnx:1.9.0'

   # Include Piper model in assets/
   ```

3. **Test Quality Gap**:
   - Compare VibeVoice (desktop) vs Piper (mobile)
   - Evaluate if quality difference is acceptable
   - Consider voice customization options

### Phase 3: Unified API (6-12 months)

**Goal**: Single codebase calling different backends

```python
# Shared interface
class UniversalTTS:
    def __init__(self, platform: str):
        if platform == "desktop":
            self.engine = VibeVoiceEngine()
        elif platform in ["ios", "android"]:
            self.engine = PiperEngine()

    def speak(self, text: str, voice: str) -> AudioData:
        return self.engine.synthesize(text, voice)

# Usage (same code everywhere)
tts = UniversalTTS(platform=detect_platform())
audio = tts.speak("Hello world", voice="en-US-female")
```

---

## Alternative: Hybrid Mobile Strategy

If Piper quality isn't enough, consider **hybrid approach**:

### On-Device (Fast, Offline)
- **Piper** via Sherpa-ONNX
- For: Quick responses, offline mode, privacy
- Quality: Good (⭐⭐⭐)

### Cloud API (High Quality, Online)
- **VibeVoice** running on your server
- For: Long-form content, narration, high-quality voice cloning
- Quality: Exceptional (⭐⭐⭐⭐⭐)
- Mobile app makes HTTP/WebSocket call to your server

```swift
// iOS hybrid approach
class HybridTTS {
    let localTTS = SherpaOnnxTTS()
    let remoteTTS = VibeVoiceAPIClient()

    func speak(text: String, quality: QualityLevel) async -> Audio {
        switch quality {
        case .fast:
            return localTTS.generate(text)  // Piper
        case .high:
            return await remoteTTS.generate(text)  // VibeVoice on server
        }
    }
}
```

**Benefits**:
- Best of both worlds
- Graceful degradation (offline → Piper)
- Your server can have multiple GPUs for scaling

---

## Cost/Benefit Analysis

### Option A: Port VibeVoice to Mobile (Universal)

| Item | Estimate |
|------|----------|
| Development time | 6-12 weeks |
| Ongoing maintenance | 1-2 days/month |
| Model size on mobile | 3-6 GB |
| Battery impact | High (GPU inference) |
| Quality | ⭐⭐⭐⭐⭐ |
| **Total Cost** | **Very High** |

### Option B: Dual-Model (VibeVoice + Piper)

| Item | Estimate |
|------|----------|
| Development time | 1-2 weeks (integration only) |
| Ongoing maintenance | Minimal (both mature projects) |
| Model size on mobile | 80-120 MB |
| Battery impact | Low (CPU/NPU inference) |
| Quality | Desktop: ⭐⭐⭐⭐⭐, Mobile: ⭐⭐⭐ |
| **Total Cost** | **Low** |

### Option C: Hybrid (Piper local + VibeVoice API)

| Item | Estimate |
|------|----------|
| Development time | 2-3 weeks |
| Ongoing maintenance | Moderate (server infrastructure) |
| Model size on mobile | 80-120 MB |
| Battery impact | Low (local) / None (API) |
| Quality | Adaptive (⭐⭐⭐ to ⭐⭐⭐⭐⭐) |
| **Total Cost** | **Moderate** |

---

## Final Recommendation

### For Your Specific Situation:

**Desktop (RTX 5090)**:
1. ✅ Use **VibeVoice-1.5B** (Python)
2. ✅ Implement model manager for loading/unloading
3. ✅ Keep Whisper + LLM resident, load VibeVoice on-demand
4. ✅ Python GStreamer element is fine

**Mobile (iOS/Android)**:
1. ✅ Use **Piper via Sherpa-ONNX**
2. ✅ Start with medium-quality models
3. ✅ Test quality gap vs VibeVoice
4. ⚠️ If gap too large → add cloud API fallback

**VRAM Management**:
```python
# Your optimal configuration (32GB VRAM)
Option A - All Resident (Recommended):
resident = {
    "whisper-large": 4GB,
    "llama-7b": 14GB,
    "vibevoice-1.5b": 7GB,
}
total = 25GB  # 7GB free for KV-cache growth

Option B - Maximum Quality:
resident = {
    "whisper-large": 4GB,
    "llama-7b": 14GB,
}
on_demand = {
    "vibevoice-7b": 15GB,  # Load when TTS needed
}
total_peak = 33GB  # Tight but workable with unloading
```

**Don't Port VibeVoice to Mobile**:
- ❌ Too much effort (6-12 weeks)
- ❌ Large model size (3-6GB)
- ❌ Battery drain
- ❌ Maintenance burden
- ✅ Piper is "good enough" for mobile
- ✅ Can always add cloud API for high-quality

---

## Code Examples

### Desktop: Model Manager with VRAM Monitoring

```python
import gc
import torch
from typing import Optional

class VRAMManager:
    """
    Manages VRAM for multi-model pipeline on RTX 5090 (32GB)
    """
    def __init__(self, max_vram_gb: float = 30.0):  # Leave 2GB headroom
        self.max_vram = max_vram_gb * 1e9
        self.models = {}
        self.resident_models = set()  # Models that stay loaded

    def get_vram_usage(self) -> float:
        """Return current VRAM usage in GB"""
        return torch.cuda.memory_allocated() / 1e9

    def get_vram_available(self) -> float:
        """Return available VRAM in GB"""
        return (self.max_vram - torch.cuda.memory_allocated()) / 1e9

    def load_resident_model(self, name: str, model_fn):
        """Load a model that stays resident"""
        print(f"Loading resident model: {name}")
        self.models[name] = model_fn()
        self.resident_models.add(name)
        print(f"VRAM usage: {self.get_vram_usage():.2f} GB / {self.max_vram/1e9:.0f} GB")

    def load_temp_model(self, name: str, model_fn):
        """Load a temporary model (will be unloaded)"""
        if name in self.models:
            return self.models[name]

        print(f"Loading temporary model: {name}")
        available = self.get_vram_available()
        print(f"Available VRAM: {available:.2f} GB")

        model = model_fn()
        self.models[name] = model
        print(f"VRAM usage: {self.get_vram_usage():.2f} GB / {self.max_vram/1e9:.0f} GB")
        return model

    def unload_temp_model(self, name: str):
        """Unload a temporary model"""
        if name in self.resident_models:
            print(f"Cannot unload resident model: {name}")
            return

        if name not in self.models:
            return

        print(f"Unloading temporary model: {name}")

        # Move to CPU first
        self.models[name].to("cpu")

        # Delete
        del self.models[name]

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(f"VRAM usage: {self.get_vram_usage():.2f} GB / {self.max_vram/1e9:.0f} GB")

    def get_model(self, name: str) -> Optional:
        """Get a loaded model"""
        return self.models.get(name)

# Usage in voice agent pipeline
class VoiceAgent:
    def __init__(self):
        self.vram = VRAMManager(max_vram_gb=30.0)  # 32GB card, 2GB headroom

        # Load resident models
        self.vram.load_resident_model("whisper", self._load_whisper)
        self.vram.load_resident_model("llm", self._load_llm)

    def _load_whisper(self):
        import whisper
        return whisper.load_model("medium", device="cuda")

    def _load_llm(self):
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-7B",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

    def _load_vibevoice(self):
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        return VibeVoiceForConditionalGenerationInference.from_pretrained(
            "vibevoice/VibeVoice-1.5B",
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio with Whisper"""
        whisper_model = self.vram.get_model("whisper")
        result = whisper_model.transcribe(audio_path)
        return result["text"]

    def chat(self, prompt: str) -> str:
        """Generate response with LLM"""
        llm_model = self.vram.get_model("llm")
        # ... generate response
        return response

    def speak(self, text: str) -> bytes:
        """Generate speech with VibeVoice"""
        # Load VibeVoice temporarily
        tts_model = self.vram.load_temp_model("vibevoice", self._load_vibevoice)

        # Generate audio
        audio = tts_model.generate(text, ...)

        # Unload immediately
        self.vram.unload_temp_model("vibevoice")

        return audio

# Example usage
agent = VoiceAgent()
# VRAM: ~18GB (whisper-medium + llama-7b)

text = agent.transcribe("input.wav")
# VRAM: ~18GB (no change)

response = agent.chat(text)
# VRAM: ~18GB (no change)

audio = agent.speak(response)
# VRAM: ~25GB (peak during TTS with VibeVoice-1.5B)
# VRAM: ~18GB (after TTS unloaded)

# Alternative: Keep all models resident (32GB allows this!)
# VRAM: ~25GB constant (whisper + llm + vibevoice all loaded)
# No loading/unloading needed! ✅
```

### Mobile iOS: Sherpa-ONNX Integration

```swift
// TTSEngine.swift
import Foundation
import SherpaOnnx

class TTSEngine {
    private var tts: SherpaOnnxOfflineTts?

    func initialize() {
        var config = sherpaOnnxOfflineTtsConfig()

        // Load Piper model from bundle
        let modelPath = Bundle.main.path(forResource: "en_US-lessac-medium", ofType: "onnx")!
        let tokensPath = Bundle.main.path(forResource: "tokens", ofType: "txt")!

        config.model.vits.model = modelPath
        config.model.vits.tokens = tokensPath
        config.model.vits.dataDir = Bundle.main.bundlePath

        // Use 2-4 threads (adjust based on device)
        config.model.numThreads = 2

        // Use CoreML for Apple Neural Engine acceleration
        config.model.provider = "coreml"

        tts = SherpaOnnxOfflineTts(config: config)
    }

    func synthesize(text: String, speed: Float = 1.0) -> [Float] {
        guard let tts = tts else {
            print("TTS not initialized")
            return []
        }

        let audio = tts.generate(text: text, sid: 0, speed: speed)
        return audio.samples
    }

    func deinitialize() {
        tts = nil
    }
}

// Usage in your app
class ViewController: UIViewController {
    let ttsEngine = TTSEngine()

    override func viewDidLoad() {
        super.viewDidLoad()
        ttsEngine.initialize()
    }

    func speakText(_ text: String) {
        let samples = ttsEngine.synthesize(text: text)
        playAudio(samples: samples, sampleRate: 22050)
    }

    func playAudio(samples: [Float], sampleRate: Int) {
        // Convert to AVAudioPCMBuffer and play
        // (Implementation details omitted)
    }
}
```

---

## Conclusion

**Your Path Forward**:

1. **Desktop (Now)**:
   - Implement VRAM manager for VibeVoice + LLM + Whisper
   - Keep Whisper + LLM resident (~16GB)
   - Load VibeVoice on-demand for TTS (~23GB peak)
   - Python GStreamer element works great on RTX 5090

2. **Mobile (Future)**:
   - Use Piper via Sherpa-ONNX
   - ~80-120MB total size
   - 50-150ms latency
   - Quality is "good enough" for mobile

3. **Don't Port VibeVoice to Mobile**:
   - Too much effort for marginal benefit
   - Piper is production-ready for mobile
   - Save VibeVoice's exceptional quality for desktop/server

**Two separate GStreamer filters/models is the right call** - optimize each platform independently rather than forcing a universal solution.
