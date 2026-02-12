import argparse
import json
import logging
import shutil
import subprocess
import threading
import time
from queue import Queue
from typing import Any, Callable, Dict, Optional

import openai
import zenoh

from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioOutputLiveStream:
    """
    A class for managing audio output and text-to-speech (TTS) conversion with OpenAI streaming.

    Parameters
    ----------
    url : str
        The URL endpoint for the text-to-speech service
    tts_model : str
        The TTS model to use for speech synthesis
    tts_voice : str
        The TTS voice to use for speech synthesis
    response_format : str, optional
        The format of the audio response (default: "pcm")
    rate : int, optional
        The sampling rate in Hz for audio output (default: 8000)
    api_key : Optional[str]
        The API key for authenticating with the TTS service
    tts_state_callback : Optional[Callable], optional
        A callback function to receive TTS state changes (active/inactive)
        (default: None)
    enable_tts_interrupt : bool, optional
        If True, enables TTS interrupt when ASR detects speech (default: False)
    extra_body : Dict[str, Any], optional
        Extra body to include in the TTS request (default: {})
    """

    def __init__(
        self,
        url: str,
        tts_model: str,
        tts_voice: str,
        response_format: str = "pcm",
        rate: int = 24000,
        api_key: Optional[str] = "",
        tts_state_callback: Optional[Callable] = None,
        enable_tts_interrupt: bool = False,
        extra_body: Dict[str, Any] = {},
    ):
        self._url = url
        self._tts_model = tts_model
        self._tts_voice = tts_voice
        self._response_format = response_format
        self._rate = rate
        self._api_key = api_key
        self._enable_tts_interrupt = enable_tts_interrupt
        self._extra_body = extra_body

        # OpenAI
        self.openai_client = openai.OpenAI(
            base_url=self._url,
            api_key=self._api_key or "no-need-api-key",
        )

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

        # Zenoh
        self.topic = "robot/status/audio"
        self.session = None
        self.pub = None
        self.audio_status = None

        try:
            self.session = open_zenoh_session()
            self.pub = self.session.declare_publisher(self.topic)
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)
            self.session.declare_subscriber("om/asr/text", self._on_asr_text)
        except Exception as e:
            logger.error(f"Failed to declare Zenoh subscriber: {e}")
            self.session = None

        # Pending requests queue
        self._pending_requests: Queue[Optional[Dict[str, str]]] = Queue()

        # Silence audio for Bluetooth optimization
        self._silence_audio = self._create_silence_audio(50)
        self._silence_prefix = self._create_silence_audio(50)

        # Running state and last audio time
        self.running: bool = True
        self._last_audio_time = time.time()

        # Persistent ffplay process for streaming
        self._ffplay_proc = None
        self._ffplay_lock = threading.RLock()
        self._ffplay_initialized = False

    def zenoh_audio_message(self, data: zenoh.Sample):
        """
        Callback function for Zenoh audio status messages.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing audio status data.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())
        if (
            self.audio_status
            and self.audio_status.sentence_to_speak.data
            and self.audio_status.status_speaker
            == AudioStatus.STATUS_SPEAKER.ACTIVE.value
        ):
            pending_message = json.loads(self.audio_status.sentence_to_speak.data)
            self.add_request(pending_message)

    def _on_asr_text(self, data: zenoh.Sample):
        """
        Callback for ASR text messages. Interrupts TTS when speech detected during playback.

        Parameters
        ----------
        data : zenoh.Sample
            The Zenoh sample containing ASR text.
        """
        try:
            if (
                self._enable_tts_interrupt
                and self.audio_status
                and self.audio_status.status_speaker
                == AudioStatus.STATUS_SPEAKER.ACTIVE.value
            ):
                asr_payload = data.payload.to_bytes()
                if asr_payload and len(asr_payload) > 0:
                    logger.debug("Interrupting TTS due to ASR detection")
                    while not self._pending_requests.empty():
                        try:
                            self._pending_requests.get_nowait()
                        except Exception as e:
                            logger.error(f"Error clearing pending TTS requests: {e}")
                            break

                    self._cleanup_ffplay()
                    logger.debug("Interrupted TTS playback")
        except Exception as e:
            logger.error(f"Error handling ASR text for interrupt: {e}")

    def set_tts_state_callback(self, callback: Callable):
        """
        Set a callback function for TTS state changes.

        Parameters
        ----------
        callback : Callable
            Function to be called when TTS state changes (active/inactive)
        """
        self._tts_state_callback = callback

    def add_request(self, audio_request: Dict[str, str]):
        """
        Add request to the TTS processing queue.

        Parameters
        ----------
        audio_request : Dict[str, str]
            Request to be processed by the TTS service
        """
        self._pending_requests.put(audio_request)

    def _process_audio(self):
        """
        Process the TTS queue and play audio output.

        Makes HTTP requests to the TTS service, converts responses to audio,
        and streams them through a persistent ffplay process.
        """
        while self.running:
            try:
                tts_request = self._pending_requests.get()
                if tts_request is None:
                    break

                if not self._initialize_ffplay():
                    logger.error("Failed to initialize ffplay")
                    continue

                self._stream_audio_chunk(self._create_silence_audio(10))

                self._tts_callback(True)
                self._update_audio_status(AudioStatus.STATUS_SPEAKER.ACTIVE.value)

                with self.openai_client.audio.speech.with_streaming_response.create(
                    model=self._tts_model,
                    voice=self._tts_voice,  # type: ignore
                    response_format=self._response_format,  # type: ignore
                    input=tts_request["text"],  # type: ignore
                    extra_body=self._extra_body,
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=1024):
                        if not self.running:
                            break
                        self._stream_audio_chunk(chunk)

                self._finish_audio_playback()

                self._tts_callback(False)
                self._update_audio_status(AudioStatus.STATUS_SPEAKER.READY.value)

            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                self._tts_callback(False)
                self._update_audio_status(AudioStatus.STATUS_SPEAKER.READY.value)
                continue

    def _initialize_ffplay(self) -> bool:
        """
        Initialize a persistent ffplay process for audio streaming.

        Returns
        -------
        bool
            True if ffplay was successfully initialized, False otherwise
        """
        with self._ffplay_lock:
            if self._ffplay_proc and self._ffplay_proc.poll() is None:
                return True  # Already running

            if not is_installed("ffplay"):
                message = (
                    "ffplay from ffmpeg not found, necessary to play audio. "
                    "On mac you can install it with 'brew install ffmpeg'. "
                    "On linux and windows you can install it from https://ffmpeg.org/"
                )
                logger.error(message)
                return False

            try:
                args = ["ffplay"]
                if "pcm" in self._response_format:
                    args.extend(["-f", "s16le", "-ar", str(self._rate)])
                args.extend(["-nodisp", "-autoexit", "-"])

                self._ffplay_proc = subprocess.Popen(
                    args=args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,  # Unbuffered for real-time streaming
                )

                self._ffplay_initialized = True
                logger.debug("Persistent ffplay process initialized")
                return True

            except Exception as e:
                logger.error(f"Failed to initialize ffplay: {e}")
                self._ffplay_proc = None
                return False

    def _stream_audio_chunk(self, chunk: bytes):
        """
        Stream an audio chunk to the persistent ffplay process.

        Parameters
        ----------
        chunk : bytes
            Audio data chunk to stream
        """
        with self._ffplay_lock:
            if self._ffplay_proc and self._ffplay_proc.poll() is None:
                try:
                    if self._ffplay_proc.stdin:
                        self._ffplay_proc.stdin.write(chunk)
                        self._ffplay_proc.stdin.flush()
                        self._last_audio_time = time.time()
                except BrokenPipeError:
                    logger.warning("ffplay process pipe broken, reinitializing")
                    self._cleanup_ffplay()
                except Exception as e:
                    logger.error(f"Error streaming audio chunk: {e}")
            else:
                logger.warning("ffplay process not available, reinitializing")
                self._cleanup_ffplay()

    def _finish_audio_playback(self):
        """
        Finish audio playback by closing stdin and waiting for ffplay to complete.
        This ensures all audio data is played before continuing.
        """
        with self._ffplay_lock:
            if self._ffplay_proc and self._ffplay_proc.poll() is None:
                try:
                    if self._ffplay_proc.stdin:
                        self._ffplay_proc.stdin.close()
                    self._ffplay_proc.wait(timeout=10)
                    logger.debug("ffplay playback finished")
                except subprocess.TimeoutExpired:
                    logger.warning("ffplay process did not finish in time, terminating")
                    self._ffplay_proc.kill()
                except Exception as e:
                    logger.error(f"Error finishing audio playback: {e}")
                finally:
                    self._ffplay_proc = None
                    self._ffplay_initialized = False

    def _cleanup_ffplay(self):
        """
        Clean up the persistent ffplay process.
        """
        with self._ffplay_lock:
            if self._ffplay_proc:
                try:
                    if self._ffplay_proc.stdin:
                        self._ffplay_proc.stdin.close()
                    self._ffplay_proc.terminate()
                    self._ffplay_proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._ffplay_proc.kill()
                except Exception as e:
                    logger.error(f"Error cleaning up ffplay: {e}")
                finally:
                    self._ffplay_proc = None
                    self._ffplay_initialized = False

    def _update_audio_status(self, speaker_status: int):
        """
        Update and publish audio status via Zenoh.

        Parameters
        ----------
        speaker_status : int
            The speaker status to set
        """
        state = AudioStatus(
            header=prepare_header(),
            status_mic=(
                self.audio_status.status_mic
                if self.audio_status
                else AudioStatus.STATUS_MIC.UNKNOWN.value
            ),
            status_speaker=speaker_status,
            sentence_to_speak=String(""),
        )

        if self.pub:
            self.pub.put(state.serialize())

    def _create_silence_audio(self, duration_ms: int = 500) -> bytes:
        """
        Create silent audio data.

        Parameters
        ----------
        duration_ms : int
            Duration of silence in milliseconds

        Returns
        -------
        bytes
            Base64 encoded silent audio data
        """
        samples = int(self._rate * duration_ms / 1000)
        silence_bytes = b"\x00" * (samples * 2)
        return silence_bytes

    def _keepalive_worker(self):
        """
        Background thread to play keepalive sounds every 60 seconds.
        """
        while self.running:
            current_time = time.time()
            if current_time - self._last_audio_time >= 60:
                self._write_audio_bytes(self._silence_audio)
                self._last_audio_time = current_time
            time.sleep(10)

    def _write_audio_bytes(self, audio_data: bytes):
        """
        Write audio data to the persistent ffplay process.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        """
        if not self._initialize_ffplay():
            logger.error("Failed to initialize ffplay for keepalive")
            return

        try:
            self._stream_audio_chunk(audio_data)
        except Exception as e:
            logger.error(f"Error writing keepalive audio: {e}")

    def _tts_callback(self, is_active: bool):
        """
        Invoke the TTS state callback if set.

        Parameters
        ----------
        is_active : bool
            Whether TTS is currently active
        """
        if self._tts_state_callback:
            self._tts_state_callback(is_active)

    def start(self):
        """
        Start the audio processing thread.

        Initializes a daemon thread for processing the TTS queue.
        """
        process_thread = threading.Thread(target=self._process_audio)
        process_thread.daemon = True
        process_thread.start()

        keepalive_thread = threading.Thread(target=self._keepalive_worker)
        keepalive_thread.daemon = True
        keepalive_thread.start()

    def run_interactive(self):
        """
        Run an interactive console for text-to-speech conversion.

        Allows users to input text for TTS conversion until 'quit' is entered
        or KeyboardInterrupt is received.
        """
        logger.info(
            "Running interactive audio output stream. Please enter text for TTS conversion."
        )
        try:
            while self.running:
                user_input = input()
                if user_input.lower() == "quit":
                    break
                self.add_request({"text": user_input})
        except KeyboardInterrupt:
            self.stop()
        finally:
            self.stop()

    def stop(self):
        """
        Stop the audio output stream and cleanup resources.
        """
        self.running = False

        # Signal shutdown to processing thread
        self._pending_requests.put(None)

        # Clean up persistent ffplay process
        self._cleanup_ffplay()

        if self.session:
            self.session.close()


def is_installed(lib_name: str) -> bool:
    """
    Check if a library is installed on the system.

    Parameters
    ----------
    lib_name : str
        The name of the library to check

    Returns
    -------
    bool
        True if the library is installed, False otherwise
    """
    return shutil.which(lib_name) is not None


def main():
    """
    Main function for running the audio output stream.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tts-url", type=str, required=True, help="URL for the TTS service"
    )
    parser.add_argument(
        "--tts-model", type=str, default="kokoro", help="TTS model to use"
    )
    parser.add_argument(
        "--tts-voice", type=str, default="af_bella", help="TTS voice to use"
    )
    parser.add_argument(
        "--response-format",
        type=str,
        default="pcm",
        help="Response format for audio output (default: pcm)",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=24000,
        help="Sampling rate for audio output (default: 24000 Hz)",
    )
    args = parser.parse_args()

    audio_output = AudioOutputLiveStream(
        args.tts_url,
        args.tts_model,
        args.tts_voice,
        rate=args.rate,
        response_format=args.response_format,
    )
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()


if __name__ == "__main__":
    main()
