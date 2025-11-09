"""
Model Manager

Manages VibeVoice model loading, unloading, and compilation.
"""
import gc
from typing import Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelManager:
    """Manages VibeVoice model loading, unloading, and compilation"""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_name = None
        self.compiled = False
        if TORCH_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"

    def load_model(self, model_name: str, compile: bool = True, diffusion_steps: int = 10, element=None):
        """Load VibeVoice model"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available. Please install torch to use VibeVoice.")

        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        # Unload existing model
        if self.model is not None:
            self.unload_model(element=element)

        print(f"Loading model: {model_name}")
        if element:
            element.emit("model-loading", model_name)

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
            self.compile_model(element=element)

        print(f"Model loaded: {model_name}")
        if element:
            element.emit("model-loaded", model_name)
            element.emit("ready", True)

    def compile_model(self, element=None):
        """Apply torch.compile optimization"""
        if not TORCH_AVAILABLE:
            return

        if self.compiled:
            return

        print("Compiling model with torch.compile...")
        if element:
            element.emit("compilation-started")

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
        if element:
            element.emit("compilation-finished")

    def unload_model(self, element=None):
        """Unload model and free VRAM"""
        if self.model is None:
            return

        print("Unloading model...")
        if element:
            element.emit("ready", False)

        # Move to CPU
        if TORCH_AVAILABLE and self.device == "cuda":
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
        if TORCH_AVAILABLE and self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model unloaded")
        if element:
            element.emit("model-unloaded")

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None

    def generate(self, text: str, voice_sample: Optional[str] = None,
                 cfg_scale: float = 1.3, **kwargs):
        """Generate audio for text"""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available")

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
