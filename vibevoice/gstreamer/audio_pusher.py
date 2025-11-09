"""
Audio Pusher Thread

Background thread that pushes audio buffers to the GStreamer src pad.
Consumes from audio_queue and pushes to srcpad with proper timestamps.
"""
import threading
import queue
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Check if numpy is available
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AudioPusher(threading.Thread):
    """
    Background thread for pushing audio buffers.

    Pulls audio chunks from audio_queue, converts to GStreamer buffers,
    sets timestamps, and pushes to src pad.
    """

    def __init__(self, audio_queue, srcpad, element, interrupt_flag):
        """
        Initialize AudioPusher thread.

        Args:
            audio_queue: Queue containing audio chunks to push
            srcpad: GStreamer src pad to push buffers to
            element: Parent GStreamer element (for sample_rate and timestamp tracking)
            interrupt_flag: threading.Event for interrupt handling
        """
        super().__init__()
        self.daemon = True  # Daemon thread exits when main thread exits
        self.name = "AudioPusher"

        self.audio_queue = audio_queue
        self.srcpad = srcpad
        self.element = element
        self.interrupt_flag = interrupt_flag

    def run(self):
        """Main thread loop - pulls audio and pushes buffers"""
        while True:
            try:
                # Get next audio chunk (blocking with timeout)
                try:
                    audio = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check interrupt flag on timeout (pusher continues regardless)
                    # Interrupt flag is for generator, not pusher
                    continue

                # None is sentinel value for EOS
                if audio is None:
                    # Send EOS event
                    eos_event = Gst.Event.new_eos()
                    self.srcpad.push_event(eos_event)
                    break

                # Convert numpy array to bytes
                if NUMPY_AVAILABLE and isinstance(audio, np.ndarray):
                    audio_bytes = audio.tobytes()
                    num_samples = len(audio)
                elif isinstance(audio, (list, tuple)):
                    # Fallback for non-numpy arrays (testing)
                    import struct
                    audio_bytes = struct.pack(f'{len(audio)}f', *audio)
                    num_samples = len(audio)
                else:
                    # Unknown type
                    continue

                # Create GStreamer buffer
                buffer = Gst.Buffer.new_wrapped(audio_bytes)

                # Calculate timestamp and duration
                sample_rate = getattr(self.element, '_sample_rate', 24000)
                current_timestamp = getattr(self.element, '_current_timestamp', 0)

                # Set PTS (presentation timestamp)
                buffer.pts = current_timestamp

                # Calculate duration in nanoseconds
                # duration = (num_samples / sample_rate) * 1_000_000_000
                duration_ns = int((num_samples / sample_rate) * 1_000_000_000)
                buffer.duration = duration_ns

                # Update timestamp for next buffer
                self.element._current_timestamp = current_timestamp + duration_ns

                # Push buffer
                ret = self.srcpad.push(buffer)

                # Handle push failure
                if ret != Gst.FlowReturn.OK:
                    # Stop pushing on error
                    break

            except Exception as e:
                # Unexpected error - stop gracefully
                break
