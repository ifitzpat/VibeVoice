"""
Tests for AudioGenerator

These tests verify the background thread that generates audio from text sentences.
"""
import pytest
import queue
import threading
import time

# Check if numpy is available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a simple array class for testing without numpy
    class np:
        @staticmethod
        def zeros(size, dtype=None):
            return [0.0] * size
        float32 = float

# Mark for skipping tests that require numpy
requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")


@pytest.fixture
def mock_model_manager():
    """Mock ModelManager for testing"""
    class MockModelManager:
        def __init__(self):
            self.is_loaded_val = True
            self.generate_called = []

        def is_loaded(self):
            return self.is_loaded_val

        def generate(self, text, voice_sample=None, cfg_scale=1.3):
            self.generate_called.append({
                'text': text,
                'voice_sample': voice_sample,
                'cfg_scale': cfg_scale
            })
            # Return 1 second of audio at 24kHz
            return np.zeros(24000, dtype=np.float32)

    return MockModelManager()


@pytest.fixture
def mock_voice_manager():
    """Mock VoiceManager for testing"""
    class MockVoiceManager:
        def __init__(self):
            self.current_voice_path = None

        def get_current_voice_path(self):
            return self.current_voice_path

    return MockVoiceManager()


@pytest.fixture
def mock_element():
    """Mock element for signal emission"""
    class MockElement:
        def __init__(self):
            self.signals = []
            self._cfg_scale = 1.3

        def emit(self, signal_name, *args):
            self.signals.append((signal_name, args))

    return MockElement()


@pytest.fixture
def audio_generator_setup(mock_model_manager, mock_voice_manager, mock_element):
    """Setup for audio generator tests"""
    sentence_queue = queue.Queue(maxsize=100)  # Large enough for tests
    audio_queue = queue.Queue(maxsize=50)  # Large enough for audio chunks
    interrupt_flag = threading.Event()

    return {
        'sentence_queue': sentence_queue,
        'audio_queue': audio_queue,
        'model_manager': mock_model_manager,
        'voice_manager': mock_voice_manager,
        'element': mock_element,
        'interrupt_flag': interrupt_flag
    }


class TestAudioGeneratorBasics:
    """Test basic AudioGenerator functionality"""

    def test_audio_generator_can_be_created(self, audio_generator_setup):
        """Test that AudioGenerator can be instantiated"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )

        assert generator is not None
        assert isinstance(generator, threading.Thread)

    def test_audio_generator_is_daemon_thread(self, audio_generator_setup):
        """Test that AudioGenerator is a daemon thread"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )

        assert generator.daemon is True


class TestAudioGeneration:
    """Test audio generation functionality"""

    def test_generator_processes_single_sentence(self, audio_generator_setup):
        """Test that generator processes a single sentence"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Put a sentence in the queue
        setup['sentence_queue'].put("Hello world")

        # Put sentinel to stop
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Check that generate was called
        assert len(setup['model_manager'].generate_called) == 1
        assert setup['model_manager'].generate_called[0]['text'] == "Hello world"

        # Check that audio was queued
        assert not setup['audio_queue'].empty()

    def test_generator_processes_multiple_sentences(self, audio_generator_setup):
        """Test that generator processes multiple sentences"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Put multiple sentences
        sentences = ["First sentence", "Second sentence", "Third sentence"]
        for sentence in sentences:
            setup['sentence_queue'].put(sentence)

        # Put sentinel
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=5.0)

        # Check all sentences were processed
        assert len(setup['model_manager'].generate_called) == 3
        for i, sentence in enumerate(sentences):
            assert setup['model_manager'].generate_called[i]['text'] == sentence

    def test_generator_uses_voice_sample_when_available(self, audio_generator_setup):
        """Test that generator uses voice sample from voice manager"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Set a voice path
        setup['voice_manager'].current_voice_path = "/path/to/voice.wav"

        # Put a sentence
        setup['sentence_queue'].put("Test")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Check voice sample was passed
        assert setup['model_manager'].generate_called[0]['voice_sample'] == "/path/to/voice.wav"

    def test_generator_uses_cfg_scale_from_element(self, audio_generator_setup):
        """Test that generator uses cfg_scale from element"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup
        setup['element']._cfg_scale = 1.5

        # Put a sentence
        setup['sentence_queue'].put("Test")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Check cfg_scale was used
        assert setup['model_manager'].generate_called[0]['cfg_scale'] == 1.5


class TestInterruptHandling:
    """Test interrupt functionality"""

    def test_generator_stops_on_interrupt_flag(self, audio_generator_setup):
        """Test that generator checks interrupt flag"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Make generation take some time
        original_generate = setup['model_manager'].generate
        call_count = [0]

        def slow_generate(*args, **kwargs):
            call_count[0] += 1
            time.sleep(0.05)  # 50ms per sentence
            return original_generate(*args, **kwargs)

        setup['model_manager'].generate = slow_generate

        # Put many sentences
        for i in range(20):
            setup['sentence_queue'].put(f"Sentence {i}")

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()

        # Wait a bit then set interrupt flag
        time.sleep(0.15)  # Allow a few sentences to process
        setup['interrupt_flag'].set()

        # Put sentinel and wait
        setup['sentence_queue'].put(None)
        generator.join(timeout=3.0)

        # Generator should have stopped early due to interrupt
        # Not all 20 sentences should be processed
        assert len(setup['model_manager'].generate_called) < 20

    def test_generator_clears_interrupt_flag_after_handling(self, audio_generator_setup):
        """Test that generator clears interrupt flag after handling"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Put a sentence
        setup['sentence_queue'].put("Test")

        # Set interrupt flag
        setup['interrupt_flag'].set()

        # Put sentinel
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Flag should be cleared
        assert not setup['interrupt_flag'].is_set()


class TestErrorHandling:
    """Test error handling"""

    def test_generator_emits_error_signal_on_exception(self, audio_generator_setup):
        """Test that generator emits error signal when generation fails"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Make model manager raise an exception
        def failing_generate(*args, **kwargs):
            raise RuntimeError("Generation failed")

        setup['model_manager'].generate = failing_generate

        # Put a sentence
        setup['sentence_queue'].put("Test")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Check error signal was emitted
        signal_names = [s[0] for s in setup['element'].signals]
        assert "generation-error" in signal_names

    def test_generator_continues_after_error(self, audio_generator_setup):
        """Test that generator continues processing after an error"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        call_count = [0]

        # Make first call fail, second succeed
        def sometimes_failing_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First call failed")
            return np.zeros(24000, dtype=np.float32)

        setup['model_manager'].generate = sometimes_failing_generate

        # Put two sentences
        setup['sentence_queue'].put("First")
        setup['sentence_queue'].put("Second")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Both calls should have been attempted
        assert call_count[0] == 2

        # Second one should have produced audio
        assert not setup['audio_queue'].empty()

    def test_generator_skips_if_model_not_loaded(self, audio_generator_setup):
        """Test that generator skips processing if model is not loaded"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup
        setup['model_manager'].is_loaded_val = False

        # Put a sentence
        setup['sentence_queue'].put("Test")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # No generation should have occurred
        assert len(setup['model_manager'].generate_called) == 0

        # Audio queue should be empty
        assert setup['audio_queue'].empty()


class TestSentinelHandling:
    """Test None sentinel handling"""

    def test_generator_stops_on_none_sentinel(self, audio_generator_setup):
        """Test that generator stops when it receives None"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Put a sentence and None
        setup['sentence_queue'].put("Test")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=2.0)

        # Thread should have stopped
        assert not generator.is_alive()

    def test_generator_processes_all_before_sentinel(self, audio_generator_setup):
        """Test that generator processes all sentences before None"""
        from vibevoice.gstreamer.audio_generator import AudioGenerator

        setup = audio_generator_setup

        # Put sentences and None
        setup['sentence_queue'].put("First")
        setup['sentence_queue'].put("Second")
        setup['sentence_queue'].put("Third")
        setup['sentence_queue'].put(None)

        # Create and start generator
        generator = AudioGenerator(
            setup['sentence_queue'],
            setup['audio_queue'],
            setup['model_manager'],
            setup['voice_manager'],
            setup['element'],
            setup['interrupt_flag']
        )
        generator.start()
        generator.join(timeout=5.0)

        # All three should have been processed
        assert len(setup['model_manager'].generate_called) == 3
