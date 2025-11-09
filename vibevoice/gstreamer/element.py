"""
GStreamer VibeVoice TTS Element

Main element implementation for VibeVoice text-to-speech.
"""
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
from gi.repository import Gst, GObject, GstBase
import threading
import queue
import time

from .audio_generator import AudioGenerator
from .audio_pusher import AudioPusher


class GstVibeVoice(Gst.Element):
    """
    GStreamer element for VibeVoice TTS.

    Pads:
        sink: Text input (text/x-raw, utf8)
        src: Audio output (audio/x-raw, F32LE, 24000Hz, mono)
        control: Control messages (application/x-vibevoice-control)
    """

    # GStreamer element metadata - must be a tuple with exactly 4 strings
    __gstmetadata__ = (
        'VibeVoice TTS',                                   # longname
        'Filter/Audio/TTS',                                # classification
        'Text-to-speech using VibeVoice',                 # description
        'Claude Code <claude@anthropic.com>'              # author
    )

    # Pad templates
    __gsttemplates__ = (
        Gst.PadTemplate.new(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "text/x-raw, format=(string)utf8; "
                "text/plain"
            )
        ),
        Gst.PadTemplate.new(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            Gst.Caps.from_string(
                "audio/x-raw, "
                "format=(string)F32LE, "
                "rate=(int)24000, "
                "channels=(int)1, "
                "layout=(string)interleaved"
            )
        ),
        Gst.PadTemplate.new(
            "control",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            Gst.Caps.from_string(
                "application/x-vibevoice-control"
            )
        )
    )

    # Signals (GObject signals for events)
    __gsignals__ = {
        "model-loading": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "model-loaded": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "model-unloaded": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "compilation-started": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "compilation-finished": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "ready": (GObject.SignalFlags.RUN_FIRST, None, (bool,)),
        "interrupted": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "voice-loaded": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "voice-changed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "queue-full": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "generation-error": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    # Properties (GObject properties for gst-inspect)
    __gproperties__ = {
        "model": (
            str,
            "Model name",
            "VibeVoice model to load",
            "vibevoice/VibeVoice-1.5B",
            GObject.ParamFlags.READWRITE
        ),
        "voice": (
            str,
            "Current voice",
            "Active voice for synthesis",
            None,
            GObject.ParamFlags.READWRITE
        ),
        "cfg-scale": (
            float,
            "CFG scale",
            "Classifier-free guidance scale",
            0.0, 5.0, 1.3,
            GObject.ParamFlags.READWRITE
        ),
        "diffusion-steps": (
            int,
            "Diffusion steps",
            "Number of diffusion steps (10-20)",
            5, 50, 10,
            GObject.ParamFlags.READWRITE
        ),
        "compile": (
            bool,
            "Use torch.compile",
            "Apply torch.compile optimization",
            True,
            GObject.ParamFlags.READWRITE
        ),
        "sentence-queue-size": (
            int,
            "Sentence queue size",
            "Maximum sentences to buffer",
            1, 1000, 100,
            GObject.ParamFlags.READWRITE
        ),
        "audio-queue-size": (
            int,
            "Audio queue size",
            "Maximum audio chunks to buffer",
            10, 500, 50,
            GObject.ParamFlags.READWRITE
        )
    }

    def __init__(self):
        super().__init__()

        # Create pads
        self.sinkpad = Gst.Pad.new_from_template(
            self.__gsttemplates__[0], "sink"
        )
        self.sinkpad.set_chain_function_full(self.chain_text)
        self.sinkpad.set_event_function_full(self.sink_event)
        self.add_pad(self.sinkpad)

        self.srcpad = Gst.Pad.new_from_template(
            self.__gsttemplates__[1], "src"
        )
        self.srcpad.set_event_function_full(self.src_event)
        self.add_pad(self.srcpad)

        # Control pad created on request
        self.controlpad = None

        # Import managers here to avoid circular imports
        from .model_manager import ModelManager
        from .voice_manager import VoiceManager

        # Managers
        self.model_manager = ModelManager()
        self.voice_manager = VoiceManager()

        # Queues
        self.sentence_queue = queue.Queue(maxsize=100)
        self.audio_queue = queue.Queue(maxsize=50)

        # Interrupt flag
        self.interrupt_flag = threading.Event()

        # Threads
        self.generator = None
        self.pusher = None
        self.running = False

        # State (use _ prefix to avoid GObject field conflicts)
        self._lock = threading.Lock()
        self._eos_received = False

        # Timing
        self._sample_rate = 24000
        self._current_timestamp = 0
        self._base_time = None

        # Properties
        self._model_name = "vibevoice/VibeVoice-1.5B"
        self._voice_name = None
        self._cfg_scale = 1.3
        self._diffusion_steps = 10
        self._use_compile = True
        self._sentence_queue_size = 100
        self._audio_queue_size = 50

        # Gst logging not available in Python bindings - use print for debugging if needed
        # print("VibeVoice element created")

    def do_request_new_pad(self, template, name, caps):
        """Handle control pad creation"""
        if template.name_template == "control":
            if self.controlpad is not None:
                # Control pad already exists
                return None

            self.controlpad = Gst.Pad.new_from_template(template, "control")
            self.controlpad.set_chain_function_full(self.chain_control)
            self.add_pad(self.controlpad)

            # Control pad created
            return self.controlpad

        return None

    def do_change_state(self, transition):
        """Handle state changes"""
        if transition == Gst.StateChange.NULL_TO_READY:
            # Initialize model (optional: can defer to first buffer)
            pass

        elif transition == Gst.StateChange.READY_TO_PAUSED:
            # Start threads
            self.start_threads()

        elif transition == Gst.StateChange.PAUSED_TO_PLAYING:
            # Resume if needed
            pass

        # Call parent
        result = Gst.Element.do_change_state(self, transition)

        if transition == Gst.StateChange.PLAYING_TO_PAUSED:
            # Pause if needed
            pass

        elif transition == Gst.StateChange.PAUSED_TO_READY:
            # Stop threads
            self.stop_threads()

        elif transition == Gst.StateChange.READY_TO_NULL:
            # Cleanup
            self.cleanup()

        return result

    def start_threads(self):
        """Start background threads"""
        with self._lock:
            self.running = True
            self._eos_received = False

            # Create and start AudioGenerator thread
            if self.generator is None or not self.generator.is_alive():
                self.generator = AudioGenerator(
                    self.sentence_queue,
                    self.audio_queue,
                    self.model_manager,
                    self.voice_manager,
                    self,
                    self.interrupt_flag
                )
                self.generator.start()

            # Create and start AudioPusher thread
            if self.pusher is None or not self.pusher.is_alive():
                self.pusher = AudioPusher(
                    self.audio_queue,
                    self.srcpad,
                    self,
                    self.interrupt_flag
                )
                self.pusher.start()

        # Threads started

    def stop_threads(self):
        """Stop background threads"""
        with self._lock:
            self.running = False

        # Send sentinel to stop generator thread
        try:
            self.sentence_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        # Wait for generator to finish (with timeout)
        if self.generator and self.generator.is_alive():
            self.generator.join(timeout=2.0)

        # Send sentinel to stop pusher thread
        try:
            self.audio_queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        # Wait for pusher to finish (with timeout)
        if self.pusher and self.pusher.is_alive():
            self.pusher.join(timeout=2.0)

        # Threads stopped

    def cleanup(self):
        """Cleanup resources"""
        self.model_manager.unload_model()
        self.voice_manager.clear()

        # Clear queues
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except queue.Empty:
                break

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def interrupt_generation(self):
        """Interrupt current generation and flush buffers"""
        with self._lock:
            # Clear sentence queue
            while not self.sentence_queue.empty():
                try:
                    self.sentence_queue.get_nowait()
                except queue.Empty:
                    break

            # Clear audio queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Set interrupt flag (generator thread checks this)
            self.interrupt_flag.set()

            # Emit interrupted signal
            self.emit("interrupted")

        # Generation interrupted, buffers flushed

    def chain_text(self, pad, parent, buffer):
        """Handle incoming text buffer"""
        # Extract text
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            # Failed to map buffer
            return Gst.FlowReturn.ERROR

        try:
            text = map_info.data.decode('utf-8').strip()
        finally:
            buffer.unmap(map_info)

        if not text:
            return Gst.FlowReturn.OK

        # Queue sentence for processing
        try:
            self.sentence_queue.put(text, block=True, timeout=1.0)
            # Queued text successfully
        except queue.Full:
            # Sentence queue full, dropping buffer
            self.emit("queue-full")
            return Gst.FlowReturn.OK  # Or ERROR to signal backpressure

        return Gst.FlowReturn.OK

    def chain_control(self, pad, parent, buffer):
        """Handle control messages"""
        # Extract JSON command
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            command_json = map_info.data.decode('utf-8')
        finally:
            buffer.unmap(map_info)

        # Process command (will be implemented in Phase 3)
        # Control command received

        return Gst.FlowReturn.OK

    def sink_event(self, pad, parent, event):
        """Handle sink pad events"""
        if event.type == Gst.EventType.EOS:
            # EOS received on sink pad
            self._eos_received = True
            # Signal generator thread
            self.sentence_queue.put(None)
            return True

        elif event.type == Gst.EventType.CAPS:
            caps = event.parse_caps()
            # Caps event received
            return True

        elif event.type == Gst.EventType.SEGMENT:
            segment = event.parse_segment()
            self._base_time = segment.time
            return True

        # Forward other events
        return self.srcpad.push_event(event)

    def src_event(self, pad, parent, event):
        """Handle src pad events"""
        # Forward upstream
        return self.sinkpad.push_event(event)

    # GObject property handlers
    def do_get_property(self, prop):
        if prop.name == "model":
            return self._model_name
        elif prop.name == "voice":
            return self._voice_name
        elif prop.name == "cfg-scale":
            return self._cfg_scale
        elif prop.name == "diffusion-steps":
            return self._diffusion_steps
        elif prop.name == "compile":
            return self._use_compile
        elif prop.name == "sentence-queue-size":
            return self._sentence_queue_size
        elif prop.name == "audio-queue-size":
            return self._audio_queue_size
        else:
            raise AttributeError(f"Unknown property {prop.name}")

    def do_set_property(self, prop, value):
        if prop.name == "model":
            self._model_name = value
        elif prop.name == "voice":
            self._voice_name = value
        elif prop.name == "cfg-scale":
            self._cfg_scale = value
        elif prop.name == "diffusion-steps":
            self._diffusion_steps = value
        elif prop.name == "compile":
            self._use_compile = value
        elif prop.name == "sentence-queue-size":
            self._sentence_queue_size = value
        elif prop.name == "audio-queue-size":
            self._audio_queue_size = value
        else:
            raise AttributeError(f"Unknown property {prop.name}")


# Don't register here - will be registered in plugin.py
