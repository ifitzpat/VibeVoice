"""
Audio Generator Thread

Background thread that generates audio from text sentences.
Consumes from sentence_queue, generates audio using VibeVoice,
and pushes audio chunks to audio_queue.
"""
import threading
import queue


class AudioGenerator(threading.Thread):
    """
    Background thread for TTS generation.

    Pulls sentences from sentence_queue, generates audio using the model manager,
    and pushes audio chunks to audio_queue.
    """

    def __init__(self, sentence_queue, audio_queue, model_manager, voice_manager, element, interrupt_flag):
        """
        Initialize AudioGenerator thread.

        Args:
            sentence_queue: Queue containing text sentences to process
            audio_queue: Queue to push generated audio chunks
            model_manager: ModelManager instance for TTS generation
            voice_manager: VoiceManager instance for voice samples
            element: Parent GStreamer element (for cfg_scale and signals)
            interrupt_flag: threading.Event for interrupt handling
        """
        super().__init__()
        self.daemon = True  # Daemon thread exits when main thread exits
        self.name = "AudioGenerator"

        self.sentence_queue = sentence_queue
        self.audio_queue = audio_queue
        self.model_manager = model_manager
        self.voice_manager = voice_manager
        self.element = element
        self.interrupt_flag = interrupt_flag

    def run(self):
        """Main thread loop - processes sentences and generates audio"""
        while True:
            try:
                # Get next sentence (blocking with timeout to check interrupt)
                try:
                    sentence = self.sentence_queue.get(timeout=0.1)
                except queue.Empty:
                    # Check interrupt flag on timeout
                    if self.interrupt_flag.is_set():
                        self.interrupt_flag.clear()
                    continue

                # None is sentinel value for EOS
                if sentence is None:
                    break

                # Check interrupt flag after getting sentence
                if self.interrupt_flag.is_set():
                    # Clear flag and skip this sentence
                    self.interrupt_flag.clear()
                    continue

                # Skip if model not loaded
                if not self.model_manager.is_loaded():
                    continue

                # Get voice sample if available
                voice_sample = self.voice_manager.get_current_voice_path()

                # Get cfg_scale from element
                cfg_scale = getattr(self.element, '_cfg_scale', 1.3)

                # Generate audio
                try:
                    audio = self.model_manager.generate(
                        text=sentence,
                        voice_sample=voice_sample,
                        cfg_scale=cfg_scale
                    )

                    # Push audio to output queue
                    if audio is not None:
                        self.audio_queue.put(audio)

                except Exception as e:
                    # Emit error signal
                    if self.element:
                        self.element.emit("generation-error", str(e))

                    # Continue processing (don't stop on single failure)
                    continue

            except Exception as e:
                # Unexpected error
                if self.element:
                    self.element.emit("generation-error", str(e))
                # Continue running
                continue
