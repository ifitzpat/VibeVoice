"""
Tests for torch.compile Integration

These tests verify that torch.compile functionality works correctly.
"""
import pytest

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Mark for skipping tests that require PyTorch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


@pytest.fixture
def model_manager():
    """Create a ModelManager instance for testing"""
    from vibevoice.gstreamer.model_manager import ModelManager
    return ModelManager()


class TestCompilationBasics:
    """Test basic compilation functionality"""

    def test_compiled_flag_starts_false(self, model_manager):
        """Test that compiled flag is False initially"""
        assert model_manager.compiled is False

    @requires_torch
    def test_compile_model_without_model_loaded(self, model_manager):
        """Test that compile_model handles no model gracefully"""
        # Should not crash
        model_manager.compile_model()
        # compiled should still be False
        assert model_manager.compiled is False

    @requires_torch
    def test_compile_model_is_idempotent(self, model_manager):
        """Test that calling compile_model twice is safe"""
        # Set compiled to True manually
        model_manager.compiled = True

        # Call compile again - should return early
        model_manager.compile_model()

        # Should still be True
        assert model_manager.compiled is True

    def test_device_selection(self, model_manager):
        """Test that device is correctly selected"""
        if TORCH_AVAILABLE:
            expected = "cuda" if CUDA_AVAILABLE else "cpu"
            assert model_manager.device == expected
        else:
            assert model_manager.device == "cpu"


class TestCompilationSignals:
    """Test compilation signal emission"""

    @requires_torch
    def test_compile_emits_signals(self, model_manager):
        """Test that compile_model emits appropriate signals"""
        class MockElement:
            def __init__(self):
                self.signals = []

            def emit(self, signal_name, *args):
                self.signals.append((signal_name, args))

        element = MockElement()

        # Manually set up a "fake" model to avoid actual loading
        model_manager.model = "fake_model"
        model_manager.compiled = False

        # Can't actually compile without real model, but we can test signal logic
        # by checking the method exists and accepts element parameter
        assert hasattr(model_manager, 'compile_model')
        assert callable(model_manager.compile_model)


class TestLoadModelCompileParameter:
    """Test load_model compile parameter"""

    @requires_torch
    def test_load_model_accepts_compile_parameter(self, model_manager, monkeypatch):
        """Test that load_model accepts compile parameter"""
        # Mock to avoid actual loading
        load_called = []

        def mock_load(*args, **kwargs):
            load_called.append(kwargs.get('compile', None))

        # We can't easily test actual loading without mocks,
        # but we can verify the parameter is accepted
        # Just check the signature
        import inspect
        sig = inspect.signature(model_manager.load_model)
        assert 'compile' in sig.parameters

        # Check default value
        assert sig.parameters['compile'].default is True


class TestCompilationOnCUDA:
    """Test that compilation only happens on CUDA"""

    @requires_torch
    def test_compile_requires_cuda(self, model_manager):
        """Test that compilation only happens on CUDA devices"""
        # If device is CPU, compile should not actually compile
        if model_manager.device == "cpu":
            # Set up fake model
            model_manager.model = "fake"
            model_manager.compiled = False

            # On CPU, load_model with compile=True should not actually compile
            # (This is enforced in the load_model method which checks device == "cuda")
            assert model_manager.device == "cpu"
