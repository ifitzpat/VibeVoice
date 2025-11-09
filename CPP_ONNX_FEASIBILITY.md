# VibeVoice C/C++ Interface & ONNX Export Feasibility Analysis

**Date**: 2025-11-09
**Question**: How Python-dependent is VibeVoice? Can it be used from C/C++? What about ONNX export?

---

## Executive Summary

**Python Dependency**: ⛔ **EXTREMELY HIGH** - Core architecture tightly coupled to Python/PyTorch ecosystem

**C/C++ Interface Options**:
1. ✅ **LibTorch** (PyTorch C++ API) - Most viable, but complex
2. ⚠️ **Python Embedding** - Works, but defeats the purpose
3. ❌ **ONNX Export** - Theoretically possible, practically infeasible for complete pipeline

**Recommendation**: If C/C++ is required, use **LibTorch with TorchScript** for component-level export, or consider **alternative TTS engines** designed for C++ deployment.

---

## Python Dependency Analysis

### Core Dependencies

From `pyproject.toml`:

```python
dependencies = [
    "torch",                    # PyTorch framework (REQUIRED)
    "transformers==4.51.3",     # Hugging Face Transformers (REQUIRED)
    "accelerate==1.6.0",        # Distributed inference (REQUIRED)
    "diffusers",                # Diffusion schedulers (REQUIRED)
    "peft",                     # LoRA fine-tuning
    "numba>=0.57.0",            # JIT compilation for DPM solver
    "llvmlite>=0.40.0",         # Numba dependency
    "scipy",                    # Signal processing
    "librosa",                  # Audio processing
    "numpy",
    "tqdm",
    "ml-collections",
    "absl-py"
]
```

### Architecture Components & Dependencies

| Component | Framework | Python-Specific Features |
|-----------|-----------|--------------------------|
| **Qwen2 LLM** | Transformers | AutoModelForCausalLM, Flash Attention, KV-cache |
| **Acoustic Tokenizer** | PyTorch | Custom VAE, depthwise convs, streaming cache |
| **Semantic Tokenizer** | PyTorch | Custom VAE, depthwise convs, streaming cache |
| **Diffusion Head** | PyTorch + Diffusers | DPMSolverMultistepScheduler, v_prediction |
| **DPM Solver** | Numba (JIT) | Python-specific JIT compilation |
| **Generation Loop** | Transformers | GenerationMixin, LogitsProcessor, complex control flow |

### Python-Specific Code Patterns

**Count of PyTorch/Transformers imports across codebase**:
```
vibevoice/modular/*.py: 68 occurrences
vibevoice/processor/*.py: Multiple
vibevoice/schedule/*.py: Numba JIT decorators
```

**Key Python-Only Features Used**:
1. **Dynamic computation graphs** (PyTorch autograd)
2. **Numba JIT compilation** (`@numba.jit` decorators in DPM solver)
3. **Transformers GenerationMixin** (complex autoregressive generation)
4. **Flash Attention 2** (CUDA kernels via PyTorch extensions)
5. **Streaming caches** (Python class-based state management)
6. **Queue-based audio streaming** (`queue.Queue`, `asyncio.Queue`)

---

## C/C++ Interfacing Options

### Option 1: LibTorch (PyTorch C++ API)

**Feasibility**: ⚠️ **POSSIBLE BUT CHALLENGING**

LibTorch is PyTorch's C++ API that can load models exported via TorchScript.

#### What Works:

```cpp
#include <torch/torch.h>
#include <torch/script.h>

// Load TorchScript model
torch::jit::script::Module model = torch::jit::load("model.pt");

// Run inference
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 100}));
auto output = model.forward(inputs).toTensor();
```

#### Export Process:

```python
# Export individual components to TorchScript
import torch

# 1. Export acoustic tokenizer decoder
acoustic_decoder = model.acoustic_tokenizer.decoder
traced_decoder = torch.jit.trace(
    acoustic_decoder,
    example_inputs=(torch.randn(1, 1, 64),)  # latent
)
traced_decoder.save("acoustic_decoder.pt")

# 2. Export diffusion head
diffusion_head = model.prediction_head
traced_diffusion = torch.jit.trace(
    diffusion_head,
    example_inputs=(
        torch.randn(1, 1, 1536),  # hidden_states
        torch.randn(1, 1, 64),    # latent
        torch.tensor([500])       # timestep
    )
)
traced_diffusion.save("diffusion_head.pt")
```

#### Critical Limitations:

**❌ Cannot Export Complete Pipeline**:
- Qwen2 LLM uses **dynamic control flow** (not TorchScript compatible)
- Generation loop has **Python-only logic** (token sampling, stopping criteria)
- DPM solver uses **Numba JIT** (Python-specific)
- Streaming caches use **Python class state** (not serializable)

**⚠️ Component-Level Export Only**:
You'd need to export each component separately and reimplement the generation loop in C++.

#### Example LibTorch Integration:

```cpp
// acoustic_decoder_inference.cpp
#include <torch/torch.h>
#include <torch/script.h>

class VibeVoiceInference {
private:
    torch::jit::script::Module acoustic_decoder;
    torch::jit::script::Module diffusion_head;
    // Note: LLM and generation loop NOT included

public:
    VibeVoiceInference(const std::string& decoder_path,
                       const std::string& diffusion_path) {
        acoustic_decoder = torch::jit::load(decoder_path);
        diffusion_head = torch::jit::load(diffusion_path);
    }

    // Decode a single latent frame to audio
    torch::Tensor decode_audio(torch::Tensor latent) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(latent);
        auto output = acoustic_decoder.forward(inputs);
        return output.toTensor();
    }

    // Run diffusion sampling (single step)
    torch::Tensor diffusion_step(torch::Tensor hidden_states,
                                  torch::Tensor latent,
                                  int64_t timestep) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(hidden_states);
        inputs.push_back(latent);
        inputs.push_back(torch::tensor({timestep}));
        return diffusion_head.forward(inputs).toTensor();
    }
};

// Usage
int main() {
    VibeVoiceInference model(
        "acoustic_decoder.pt",
        "diffusion_head.pt"
    );

    // You'd still need to:
    // 1. Run Qwen2 LLM (separate process or LibTorch)
    // 2. Implement DPM solver in C++
    // 3. Implement generation loop in C++
    // 4. Handle text tokenization
    // 5. Manage KV-caches
}
```

**Verdict**: You can export **individual neural network components**, but you'd need to **rewrite the entire generation pipeline in C++** (1000+ lines of complex logic).

---

### Option 2: Python Embedding via CPython API

**Feasibility**: ✅ **WORKS, BUT DEFEATS PURPOSE**

Embed Python interpreter in C++ and call VibeVoice Python API.

```cpp
#include <Python.h>

int main() {
    Py_Initialize();

    PyRun_SimpleString(
        "from vibevoice.modular.modeling_vibevoice_inference import *\n"
        "from vibevoice.processor.vibevoice_processor import *\n"
        "model = VibeVoiceForConditionalGenerationInference.from_pretrained(...)\n"
    );

    // Call Python functions from C++
    PyObject* pModule = PyImport_ImportModule("vibevoice.modular.modeling_vibevoice_inference");
    // ... etc

    Py_Finalize();
}
```

**Pros**:
- Works immediately
- No model export needed
- Full functionality

**Cons**:
- Still requires Python runtime
- GIL (Global Interpreter Lock) overhead
- Memory management complexity
- Defeats the purpose of C++ interface

**Verdict**: This is essentially what the GStreamer Python element does. Not a true C++ solution.

---

### Option 3: ONNX Export

**Feasibility**: ❌ **THEORETICALLY POSSIBLE, PRACTICALLY INFEASIBLE**

#### Why ONNX Export is Challenging:

**1. Qwen2 LLM ONNX Export**

Qwen2 can be exported using Hugging Face Optimum:

```bash
pip install optimum[exporters]

optimum-cli export onnx \
  --model Qwen/Qwen2-1.5B \
  --task text-generation-with-past \
  qwen2_onnx/
```

**Problems**:
- Export works for **standard inference**, not **custom generation loops**
- VibeVoice uses a **hybrid AR + diffusion** generation with custom token constraints
- Flash Attention 2 not ONNX-compatible (falls back to slower attention)
- KV-cache management differs from standard transformers

**2. Custom Components (Tokenizers, Diffusion Head)**

Can be exported with `torch.onnx.export`:

```python
import torch.onnx

# Export acoustic decoder
torch.onnx.export(
    model.acoustic_tokenizer.decoder,
    (torch.randn(1, 1, 64),),  # example input
    "acoustic_decoder.onnx",
    input_names=["latent"],
    output_names=["audio"],
    dynamic_axes={
        "latent": {0: "batch", 1: "frames"},
        "audio": {0: "batch", 1: "samples"}
    }
)
```

**Problems**:
- **Streaming caches**: ONNX has no concept of stateful caches (PyTorch-specific)
- **Numba JIT code**: DPM solver cannot be exported (Python runtime dependency)
- **Custom operators**: ConvRMSNorm, fused operations may not translate cleanly

**3. Complete Generation Pipeline**

The generation loop (`generate()` method) includes:

```python
for step in range(max_steps):
    # LLM forward pass
    outputs = self.model(**model_inputs)

    # Token sampling with constraints
    next_token = self._sample_next_token(outputs.logits)

    # Conditional diffusion sampling
    if next_token == speech_diffusion_id:
        latent = self.sample_speech_tokens(
            positive_condition, negative_condition, cfg_scale
        )
        audio = self.acoustic_tokenizer.decode(latent, cache=cache)
        streamer.put(audio)

    # Update caches and embeddings
    # ... complex state management
```

**ONNX Cannot Handle**:
- Dynamic control flow (if/else based on token values)
- Loops with variable iteration counts
- External state (caches, queues)
- Python callbacks (streamer.put, stop_check_fn)

#### Partial ONNX Export Strategy

You could export **individual components** and reassemble in C++ with ONNX Runtime:

```cpp
#include <onnxruntime_cxx_api.h>

class VibeVoiceONNX {
private:
    Ort::Session qwen2_session;
    Ort::Session acoustic_decoder_session;
    Ort::Session diffusion_head_session;

public:
    // Load ONNX models
    VibeVoiceONNX() {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VibeVoice");
        Ort::SessionOptions opts;

        qwen2_session = Ort::Session(env, "qwen2.onnx", opts);
        acoustic_decoder_session = Ort::Session(env, "decoder.onnx", opts);
        diffusion_head_session = Ort::Session(env, "diffusion.onnx", opts);
    }

    // Inference (simplified)
    std::vector<float> generate_audio(const std::string& text) {
        // 1. Tokenize text (need external library)
        // 2. Run Qwen2 forward (ONNX Runtime)
        // 3. Sample tokens (implement in C++)
        // 4. Run diffusion (ONNX Runtime)
        // 5. Decode audio (ONNX Runtime)
        // 6. Implement DPM solver from scratch in C++
        // 7. Manage streaming caches manually
    }
};
```

**Effort Required**:
- ✅ Export 3-4 neural network components (moderate difficulty)
- ❌ Reimplement generation loop in C++ (1000+ lines, high complexity)
- ❌ Reimplement DPM solver in C++ (100+ lines, mathematical complexity)
- ❌ Implement streaming cache logic (200+ lines)
- ❌ Text tokenization (need external library like SentencePiece)

**Verdict**: Possible for an experienced team with 2-4 weeks of effort, but high risk of bugs and maintenance burden.

---

## Practical Recommendations

### Recommendation 1: Stay with Python

**Best Option for Most Users**

Use Python with GStreamer bindings (as outlined in STREAMING_TTS_ASSESSMENT.md).

**Pros**:
- Works immediately
- Full feature support
- Easy to debug and maintain
- Community support

**Cons**:
- Python runtime overhead
- GIL contention in multi-threaded pipelines

### Recommendation 2: LibTorch for Performance-Critical Paths

**For Advanced Users**

Export individual components to LibTorch and implement generation loop in C++.

**Effort**: 2-4 weeks for experienced C++ developer
**Use Case**: Production systems requiring maximum performance

**Steps**:
1. Export acoustic/semantic decoders via TorchScript
2. Export diffusion head via TorchScript
3. Load Qwen2 via LibTorch (or separate ONNX Runtime)
4. Reimplement generation loop in C++
5. Port DPM solver to C++ (translate from Python/Numba)

### Recommendation 3: Alternative TTS Engines for C++

**If C++ is a hard requirement, consider TTS engines designed for C++ deployment:**

| Engine | Language | ONNX Support | Quality | Latency |
|--------|----------|--------------|---------|---------|
| **Piper** | C++ (ONNX Runtime) | ✅ Native | Good | 50-150ms |
| **Sherpa-ONNX** | C++ | ✅ Native | Good | 100ms |
| **espeak-ng** | C | ✅ N/A | Basic | <10ms |
| **MaryTTS** | Java | ❌ | Good | 500ms |

**Piper** is the best alternative - it's designed from the ground up for C++ deployment with ONNX.

#### Piper C++ Usage:

```cpp
#include "piper.hpp"

int main() {
    piper::PiperConfig config;
    piper::Voice voice;

    piper::loadVoice(config, "en_US-lessac-medium.onnx",
                     "en_US-lessac-medium.onnx.json", voice);

    std::vector<int16_t> audio;
    piper::textToAudio(config, voice, "Hello world", audio);

    // Write audio to GStreamer buffer
}
```

### Recommendation 4: Hybrid Approach

**Best of Both Worlds**

- **VibeVoice (Python)**: High-quality, non-interactive content
- **Piper (C++)**: Low-latency, interactive responses
- **GStreamer Router**: Direct traffic based on priority

---

## Technical Deep Dive: Why ONNX Export is Hard

### Problem 1: Dynamic Control Flow

ONNX supports static graphs. VibeVoice's generation loop has **dynamic control flow**:

```python
# This cannot be directly exported to ONNX
for step in range(max_steps):  # Variable iteration count
    next_token = sample_token(logits)

    if next_token == speech_diffusion_id:  # Conditional branching
        # Run diffusion
    elif next_token == speech_end_id:
        break  # Early exit
```

**ONNX Limitation**: Requires graph to be known at export time.

**Workaround**: Export inner functions, implement loop in C++.

### Problem 2: Stateful Caches

VibeVoice uses streaming caches for tokenizers:

```python
class VibeVoiceTokenizerStreamingCache:
    def __init__(self):
        self.cache = {}  # Per-sample state

    def get(self, sample_idx):
        return self.cache.get(sample_idx, None)
```

**ONNX Limitation**: No concept of persistent state across calls.

**Workaround**: Manage cache externally in C++, pass as input/output tensors.

### Problem 3: Numba JIT Code

DPM solver uses Numba for performance:

```python
@numba.jit(nopython=True)
def dpm_solver_update(model_output, sample, noise_pred_prev, t, ...):
    # Complex numerical solver
    return updated_sample
```

**ONNX Limitation**: Cannot export JIT-compiled Python code.

**Workaround**: Rewrite in C++ from mathematical specification.

### Problem 4: Python Callbacks

Generation loop uses Python callbacks:

```python
outputs = model.generate(
    ...,
    audio_streamer=streamer,  # Python object
    stop_check_fn=lambda: user_stopped,  # Python lambda
    tqdm_class=custom_tqdm  # Python class
)
```

**ONNX Limitation**: No support for callbacks.

**Workaround**: Implement callback mechanism in C++.

---

## Estimation: Full C++ Port Effort

| Task | Lines of Code | Complexity | Time (Experienced Dev) |
|------|---------------|------------|------------------------|
| Export neural networks to ONNX/TorchScript | - | Medium | 1-2 days |
| C++ ONNX Runtime integration | 200 | Medium | 2-3 days |
| Reimplement generation loop | 800 | High | 1 week |
| Port DPM solver | 150 | High | 2-3 days |
| Implement streaming cache logic | 200 | Medium | 2-3 days |
| Text tokenization integration | 100 | Medium | 1-2 days |
| Testing and debugging | - | High | 1 week |
| **Total** | **~1500** | **High** | **3-4 weeks** |

**Risk Factors**:
- Numerical accuracy differences (C++ vs Python)
- Memory leak potential in manual cache management
- Undefined behavior from tensor shape mismatches
- Maintenance burden for future model updates

---

## Conclusion

### Can VibeVoice be used from C/C++?

**Short Answer**: Not easily.

**Realistic Options**:
1. **Python embedding** (works but defeats purpose)
2. **LibTorch component export + C++ generation loop** (feasible but 3-4 weeks effort)
3. **Full ONNX export** (theoretically possible, practically infeasible)

### Should You Port to C++?

**Consider C++ port if**:
- ✅ You have experienced C++ team (2+ people)
- ✅ You need absolute maximum performance
- ✅ You're willing to invest 4-6 weeks
- ✅ You can maintain custom C++ codebase

**Stick with Python if**:
- ✅ Time-to-market is important
- ✅ Team is primarily Python-focused
- ✅ Python performance is "good enough" (it will be on RTX 5090)
- ✅ You want to leverage future VibeVoice updates

### Best Path Forward

**For GStreamer integration**: Use **Python GStreamer element** as recommended in STREAMING_TTS_ASSESSMENT.md.

**If C++ is absolutely required**: Consider **Piper TTS** instead - it's designed for C++ deployment from day one.

**If you must use VibeVoice in C++**: Budget 4-6 weeks for a LibTorch-based partial port, focusing on exporting the neural network components and reimplementing the generation logic.

---

## Appendix: Alternative Approach - Separate Service

Instead of embedding in C++, run VibeVoice as a **separate service**:

```
┌─────────────────┐         gRPC/WebSocket          ┌──────────────┐
│  C++ GStreamer  │ ───────────────────────────────→│ VibeVoice    │
│  Pipeline       │                                  │ Python       │
│                 │ ←───────────────────────────────│ Service      │
└─────────────────┘        Audio chunks             └──────────────┘
```

**Pros**:
- Clean separation of concerns
- Python service can scale independently
- Can restart Python service without pipeline disruption
- Easier to develop and debug

**Cons**:
- Network/IPC latency overhead (~1-5ms)
- Additional infrastructure complexity

This might be the **pragmatic sweet spot** for production systems.
