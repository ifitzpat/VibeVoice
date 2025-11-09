"""
Tests for AudioPusher

These tests verify the background thread that pushes audio buffers to the src pad.
"""
import pytest
import queue
import threading
import time
import gi

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
        @staticmethod
        def array(data, dtype=None):
            return data
        float32 = float

# Mark for skipping tests that require numpy
requires_numpy = pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not installed")

gi.require_version('Gst', '1.0')
from gi.repository import Gst


@pytest.fixture
def mock_srcpad():
    """Mock src pad for testing"""
    class MockSrcPad:
        def __init__(self):
            self.pushed_buffers = []
            self.pushed_events = []

        def push(self, buffer):
            # Extract data from buffer for testing
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                data = bytes(map_info.data)
                buffer.unmap(map_info)
                self.pushed_buffers.append({
                    'data': data,
                    'pts': buffer.pts,
                    'duration': buffer.duration,
                    'size': len(data)
                })
            return Gst.FlowReturn.OK

        def push_event(self, event):
            self.pushed_events.append(event)
            return True

    return MockSrcPad()


@pytest.fixture
def mock_element():
    """Mock element for pusher"""
    class MockElement:
        def __init__(self):
            self._sample_rate = 24000
            self._current_timestamp = 0

    return MockElement()


@pytest.fixture
def audio_pusher_setup(mock_srcpad, mock_element):
    """Setup for audio pusher tests"""
    audio_queue = queue.Queue(maxsize=50)  # Large enough for audio chunks
    interrupt_flag = threading.Event()

    return {
        'audio_queue': audio_queue,
        'srcpad': mock_srcpad,
        'element': mock_element,
        'interrupt_flag': interrupt_flag
    }


class TestAudioPusherBasics:
    """Test basic AudioPusher functionality"""

    def test_audio_pusher_can_be_created(self, audio_pusher_setup):
        """Test that AudioPusher can be instantiated"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )

        assert pusher is not None
        assert isinstance(pusher, threading.Thread)

    def test_audio_pusher_is_daemon_thread(self, audio_pusher_setup):
        """Test that AudioPusher is a daemon thread"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )

        assert pusher.daemon is True


class TestAudioPushing:
    """Test audio pushing functionality"""

    def test_pusher_pushes_single_audio_chunk(self, audio_pusher_setup):
        """Test that pusher pushes a single audio chunk"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunk (1 second at 24kHz)
        audio_chunk = np.zeros(24000, dtype=np.float32)
        setup['audio_queue'].put(audio_chunk)

        # Put sentinel to stop
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check that buffer was pushed
        assert len(setup['srcpad'].pushed_buffers) == 1

    def test_pusher_pushes_multiple_audio_chunks(self, audio_pusher_setup):
        """Test that pusher pushes multiple audio chunks"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put multiple audio chunks
        num_chunks = 5
        for i in range(num_chunks):
            audio_chunk = np.zeros(24000, dtype=np.float32)
            setup['audio_queue'].put(audio_chunk)

        # Put sentinel
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=3.0)

        # Check all chunks were pushed
        assert len(setup['srcpad'].pushed_buffers) == num_chunks

    def test_pusher_converts_float32_to_bytes(self, audio_pusher_setup):
        """Test that pusher converts float32 numpy array to bytes"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunk with known values
        audio_chunk = np.array([0.5, -0.5, 0.0, 1.0], dtype=np.float32)
        setup['audio_queue'].put(audio_chunk)
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check buffer size matches (4 samples * 4 bytes per float32)
        assert setup['srcpad'].pushed_buffers[0]['size'] == 16


class TestTimestamping:
    """Test timestamp calculation"""

    def test_pusher_sets_timestamps_on_buffers(self, audio_pusher_setup):
        """Test that pusher sets PTS on buffers"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunk
        audio_chunk = np.zeros(24000, dtype=np.float32)
        setup['audio_queue'].put(audio_chunk)
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check PTS is set
        assert setup['srcpad'].pushed_buffers[0]['pts'] is not None
        assert setup['srcpad'].pushed_buffers[0]['pts'] >= 0

    def test_pusher_increments_timestamps(self, audio_pusher_setup):
        """Test that pusher increments timestamps for consecutive buffers"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put multiple audio chunks
        for i in range(3):
            audio_chunk = np.zeros(24000, dtype=np.float32)  # 1 second each
            setup['audio_queue'].put(audio_chunk)

        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=3.0)

        # Check timestamps are incrementing
        buffers = setup['srcpad'].pushed_buffers
        assert len(buffers) == 3
        assert buffers[1]['pts'] > buffers[0]['pts']
        assert buffers[2]['pts'] > buffers[1]['pts']

    def test_pusher_sets_duration_on_buffers(self, audio_pusher_setup):
        """Test that pusher sets duration on buffers"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunk (1 second at 24kHz)
        audio_chunk = np.zeros(24000, dtype=np.float32)
        setup['audio_queue'].put(audio_chunk)
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check duration is set (should be 1 second in nanoseconds)
        # 1 second = 1,000,000,000 nanoseconds
        duration_ns = setup['srcpad'].pushed_buffers[0]['duration']
        assert duration_ns is not None
        # Should be approximately 1 second (allow some tolerance)
        assert 900_000_000 <= duration_ns <= 1_100_000_000

    def test_pusher_calculates_duration_correctly_for_different_sizes(self, audio_pusher_setup):
        """Test that pusher calculates duration correctly for different audio sizes"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunks of different sizes
        # 24000 samples = 1 second at 24kHz
        # 12000 samples = 0.5 seconds at 24kHz
        setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))
        setup['audio_queue'].put(np.zeros(12000, dtype=np.float32))
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check durations
        buffers = setup['srcpad'].pushed_buffers
        # Second buffer should have half the duration of first
        assert buffers[1]['duration'] < buffers[0]['duration']
        # Should be approximately half
        ratio = buffers[1]['duration'] / buffers[0]['duration']
        assert 0.45 <= ratio <= 0.55


class TestInterruptHandling:
    """Test interrupt functionality"""

    def test_pusher_respects_interrupt_flag(self, audio_pusher_setup):
        """Test that pusher checks interrupt flag during operation"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Set interrupt flag before starting
        setup['interrupt_flag'].set()

        # Put some audio chunks
        for i in range(5):
            setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))

        # Put sentinel
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Pusher should complete (interrupt doesn't stop pusher, just generator)
        # All 5 chunks should still be pushed
        assert len(setup['srcpad'].pushed_buffers) == 5


class TestSentinelHandling:
    """Test None sentinel handling"""

    def test_pusher_stops_on_none_sentinel(self, audio_pusher_setup):
        """Test that pusher stops when it receives None"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio chunk and None
        setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Thread should have stopped
        assert not pusher.is_alive()

    def test_pusher_pushes_all_before_sentinel(self, audio_pusher_setup):
        """Test that pusher pushes all chunks before None"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put chunks and None
        for i in range(3):
            setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=3.0)

        # All three should have been pushed
        assert len(setup['srcpad'].pushed_buffers) == 3

    def test_pusher_sends_eos_after_sentinel(self, audio_pusher_setup):
        """Test that pusher sends EOS event after None sentinel"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Put audio and None
        setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Check EOS event was sent
        assert len(setup['srcpad'].pushed_events) > 0
        # Last event should be EOS
        eos_event = setup['srcpad'].pushed_events[-1]
        assert eos_event.type == Gst.EventType.EOS


class TestErrorHandling:
    """Test error handling"""

    def test_pusher_handles_push_failure_gracefully(self, audio_pusher_setup):
        """Test that pusher handles push failures gracefully"""
        from vibevoice.gstreamer.audio_pusher import AudioPusher

        setup = audio_pusher_setup

        # Make srcpad.push fail
        def failing_push(buffer):
            return Gst.FlowReturn.ERROR

        setup['srcpad'].push = failing_push

        # Put audio chunks
        setup['audio_queue'].put(np.zeros(24000, dtype=np.float32))
        setup['audio_queue'].put(None)

        # Create and start pusher
        pusher = AudioPusher(
            setup['audio_queue'],
            setup['srcpad'],
            setup['element'],
            setup['interrupt_flag']
        )
        pusher.start()
        pusher.join(timeout=2.0)

        # Pusher should have stopped gracefully (not crashed)
        assert not pusher.is_alive()
