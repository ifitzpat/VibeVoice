"""
Tests for ModelManager

These tests verify model loading, unloading, compilation, and generation.
"""
import pytest

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Mark for skipping tests that require PyTorch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


@pytest.fixture
def model_manager():
    """Create a ModelManager instance for testing"""
    from vibevoice.gstreamer.model_manager import ModelManager

    return ModelManager()


class TestModelManagerBasics:
    """Test basic ModelManager functionality"""

    def test_model_manager_initialization(self, model_manager):
        """Test that ModelManager initializes correctly"""
        assert model_manager.model is None
        assert model_manager.processor is None
        assert model_manager.model_name is None
        assert model_manager.compiled is False
        assert model_manager.device in ["cuda", "cpu"]

    def test_is_loaded_returns_false_initially(self, model_manager):
        """Test that is_loaded returns False when no model is loaded"""
        assert model_manager.is_loaded() is False


class TestModelLoading:
    """Test model loading functionality"""

    @requires_torch
    def test_load_model_sets_model_name(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that load_model sets the model name"""
        # Mock the imports
        def mock_from_pretrained_model(*args, **kwargs):
            return mock_model

        def mock_from_pretrained_processor(*args, **kwargs):
            return mock_processor

        # Patch the from_pretrained methods
        import vibevoice.gstreamer.model_manager as mm
        monkeypatch.setattr(
            "vibevoice.modular.modeling_vibevoice_inference.VibeVoiceForConditionalGenerationInference.from_pretrained",
            mock_from_pretrained_model,
            raising=False
        )
        monkeypatch.setattr(
            "vibevoice.processor.vibevoice_processor.VibeVoiceProcessor.from_pretrained",
            mock_from_pretrained_processor,
            raising=False
        )

        # Mock the module imports
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        # Set up the mocks
        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False)

        assert model_manager.model is not None
        assert model_manager.processor is not None
        assert model_manager.model_name == "test-model"
        assert model_manager.is_loaded() is True

    @requires_torch
    def test_load_model_emits_signals(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that load_model emits appropriate signals"""
        # Setup mocks
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        # Mock element to capture signals
        class MockElement:
            def __init__(self):
                self.signals = []

            def emit(self, signal_name, *args):
                self.signals.append((signal_name, args))

        element = MockElement()
        model_manager.load_model("test-model", compile=False, element=element)

        # Check signals
        signal_names = [s[0] for s in element.signals]
        assert "model-loading" in signal_names
        assert "model-loaded" in signal_names
        assert "ready" in signal_names

    @requires_torch
    def test_load_model_sets_diffusion_steps(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that load_model sets diffusion steps on model"""
        # Setup mocks
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False, diffusion_steps=15)

        assert hasattr(model_manager.model, 'steps')
        assert model_manager.model.steps == 15


class TestModelUnloading:
    """Test model unloading functionality"""

    @requires_torch
    def test_unload_model_clears_state(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that unload_model clears all model state"""
        # Load a model first
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False)
        assert model_manager.is_loaded()

        # Unload
        model_manager.unload_model()

        assert model_manager.model is None
        assert model_manager.processor is None
        assert model_manager.model_name is None
        assert model_manager.compiled is False
        assert model_manager.is_loaded() is False

    @requires_torch
    def test_unload_model_emits_signals(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that unload_model emits appropriate signals"""
        # Load a model first
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False)

        # Mock element
        class MockElement:
            def __init__(self):
                self.signals = []

            def emit(self, signal_name, *args):
                self.signals.append((signal_name, args))

        element = MockElement()
        model_manager.unload_model(element=element)

        signal_names = [s[0] for s in element.signals]
        assert "ready" in signal_names
        assert "model-unloaded" in signal_names

        # Check that ready signal has False
        ready_signals = [s for s in element.signals if s[0] == "ready"]
        assert ready_signals[0][1] == (False,)

    def test_unload_model_does_nothing_if_not_loaded(self, model_manager):
        """Test that unload_model is safe to call when no model is loaded"""
        # Should not raise an error
        model_manager.unload_model()
        assert model_manager.is_loaded() is False


class TestModelGeneration:
    """Test model generation functionality"""

    def test_generate_raises_if_model_not_loaded(self, model_manager):
        """Test that generate raises error if model is not loaded"""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model_manager.generate("test text")

    @requires_torch
    def test_generate_returns_audio(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that generate returns audio data"""
        # Load model
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False)

        # Generate
        audio = model_manager.generate("test text")

        assert audio is not None
        assert len(audio) > 0

    @requires_torch
    def test_generate_uses_voice_sample(self, model_manager, monkeypatch, mock_model, mock_processor):
        """Test that generate can use voice samples"""
        # Load model
        class MockInference:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_model

        class MockProcessor:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return mock_processor

        import sys
        sys.modules['vibevoice.modular.modeling_vibevoice_inference'] = type('Module', (), {
            'VibeVoiceForConditionalGenerationInference': MockInference
        })()
        sys.modules['vibevoice.processor.vibevoice_processor'] = type('Module', (), {
            'VibeVoiceProcessor': MockProcessor
        })()

        model_manager.load_model("test-model", compile=False)

        # Generate with voice sample
        audio = model_manager.generate("test text", voice_sample="/path/to/voice.wav")

        assert audio is not None
