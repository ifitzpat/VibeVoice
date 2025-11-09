"""
Pytest configuration and fixtures for GStreamer VibeVoice tests
"""
import pytest
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst


@pytest.fixture(scope="session", autouse=True)
def gst_init():
    """Initialize GStreamer once for all tests"""
    Gst.init(None)
    yield
    Gst.deinit()


@pytest.fixture
def mock_model():
    """Mock VibeVoice model for testing"""
    class MockModel:
        def __init__(self):
            self.device = "cpu"
            self.compiled = False

        def eval(self):
            return self

        def to(self, device):
            self.device = device
            return self

        def set_ddpm_inference_steps(self, steps):
            self.steps = steps

        def generate(self, **kwargs):
            # Return mock output
            import numpy as np
            class MockOutput:
                def __init__(self):
                    # Generate 1 second of silence at 24kHz
                    self.speech_outputs = [[np.zeros(24000, dtype=np.float32)]]
            return MockOutput()

    return MockModel()


@pytest.fixture
def mock_processor():
    """Mock VibeVoice processor for testing"""
    class MockProcessor:
        def __init__(self):
            self.tokenizer = None

        def __call__(self, text, voice_samples, return_tensors):
            # Return mock inputs
            class MockInputs:
                def to(self, device):
                    return self
            return MockInputs()

    return MockProcessor()
