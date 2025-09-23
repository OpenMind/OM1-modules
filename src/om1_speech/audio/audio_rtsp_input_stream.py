import base64
import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from queue import Empty, Full
from typing import Callable, Dict, Generator, List, Optional, Union

import av
import numpy as np

from om1_utils import LoggingConfig, get_logging_config, setup_logging
from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


def rtsp_audio_processor(
    rtsp_url: str,
    rate: int,
    chunk: int,
    audio_data_queue: mp.Queue,
    control_queue: mp.Queue,
    logging_config: Optional[LoggingConfig] = None,
):
    """
    Process RTSP audio stream and put audio data into a multiprocessing queue.

    This function runs in a separate process to handle RTSP audio streaming,
    resampling, and chunking. It puts the processed audio data into a
    multiprocessing queue for consumption by other processes or threads.

    Parameters
    ----------
    rtsp_url : str
        The RTSP URL of the audio stream
    rate : int
        The sampling rate in Hz for audio capture
    chunk : int
        The size of each audio chunk in frames
    audio_data_queue : mp.Queue
        A multiprocessing queue to put audio data chunks.
    control_queue : mp.Queue
        A multiprocessing queue to receive control commands (e.g., stop).
    """
    setup_logging("rtsp_audio_processor", logging_config=logging_config)

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=rate,
    )

    running = True
    while running:
        try:
            cmd = control_queue.get_nowait()
            if cmd == "STOP":
                running = False
                break
        except Empty:
            pass

        try:
            rtsp_container = av.open(rtsp_url, options={"rtsp_transport": "tcp"})

            if not rtsp_container.streams.audio:
                logging.error("No audio stream found in RTSP. Retrying...")
                time.sleep(2.0)
                continue

            rtsp_audio_stream = rtsp_container.streams.audio[0]
            logging.info("RTSP audio stream connected successfully")

            audio_buffer = np.array([], dtype=np.int16)
            for frame in rtsp_container.decode(rtsp_audio_stream):
                if not running:
                    break

                frame = resampler.resample(frame)[0]
                frame_data = frame.to_ndarray().astype(np.int16).flatten()
                audio_buffer = np.concatenate([audio_buffer, frame_data])

                if len(audio_buffer) >= chunk:
                    chunk_data = audio_buffer[:chunk]
                    audio_buffer = audio_buffer[chunk:]

                    pcm = chunk_data.tobytes()
                    try:
                        audio_data_queue.put(pcm, timeout=1.0)
                    except Full:
                        logging.warning("Audio data queue is full, dropping chunk")

        except Exception as e:
            logging.error(f"RTSP audio error: {e}")
            if rtsp_container:
                rtsp_container.close()
                rtsp_container = None

            if running:
                logging.info("Reconnecting in 2 seconds...")
                time.sleep(2.0)

        finally:
            if rtsp_container:
                rtsp_container.close()
                rtsp_container = None

        time.sleep(0.01)


class AudioRTSPInputStream:
    """
    A class for capturing and managing real-time audio input from RTSP streams.

    This class provides functionality to capture audio data from RTSP streams,
    process it in chunks, and make it available through a generator or callback
    mechanism. It supports Text-to-Speech (TTS) integration by temporarily
    suspending audio capture during TTS playback.

    Parameters
    ----------
    rate : int, optional
        The sampling rate in Hz for audio capture. If None, uses the rate from RTSP stream.
        (default: 16000)
    chunk : int, optional
        The size of each audio chunk in frames. If None, automatically calculates an optimal chunk size
        based on the sample rate (approximately 200ms of audio).
        (default: None)
    audio_data_callback : Optional[Callable], optional
        A callback function that receives audio data chunks (default: None)
    language_code: str, optional
        The language for the ASR to listen. (default: en-US)
    rtsp_url : str, optional
        The RTSP URL of the audio stream. If None, uses device or device_name to determine the URL.
        (default: "rtsp://localhost:8554/live")
    """

    def __init__(
        self,
        rtsp_url: str = "rtsp://localhost:8554/live",
        rate: int = 16000,
        chunk: Optional[int] = None,
        audio_data_callback: Optional[Callable] = None,
        audio_data_callbacks: Optional[List[Callable]] = None,
        language_code: Optional[str] = None,
    ):
        self._rate = rate

        # Determine RTSP URL from device parameters
        self._rtsp_url = rtsp_url

        # Determine language code
        if language_code is None:
            self._language_code = "en-US"
            logger.info(f"Using default language code: {self._language_code}")
        else:
            self._language_code = language_code
            logger.info(f"Using specified language code: {self._language_code}")

        # Callback for audio data
        self._audio_data_callbacks = audio_data_callbacks or []
        self.register_audio_data_callback(audio_data_callback)

        # Flag to indicate if TTS is active
        self._is_tts_active: bool = False

        # Thread-safe buffer for audio data
        self._buff = mp.Queue()
        self._control_queue = mp.Queue()

        # Audio processing thread
        self._rtsp_audio_processor_thread: Optional[threading.Thread] = None

        # Audio callback thread
        self._audio_callback_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        # Zenoh
        self.topic = "robot/status/audio"
        self.session = None
        self.pub = None
        self.audio_status = None

        try:
            self.session = open_zenoh_session()
            self.pub = self.session.declare_publisher(self.topic)
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)
        except Exception as e:
            logger.error(f"Failed to declare Zenoh subscriber: {e}")
            self.session = None

        self.running: bool = True

        if chunk is None:
            self._chunk = int(rate * 0.5)
            logger.info(f"Auto-calculated chunk size: {self._chunk} frames")
        else:
            self._chunk = chunk
            chunk_duration_ms = (self._chunk / self._rate) * 1000
            logger.info(
                f"Using specified chunk size: {self._chunk} frames ({chunk_duration_ms:.2f} ms)"
            )

    def zenoh_audio_message(self, data: str):
        """
        Callback function for Zenoh audio status messages.

        Parameters
        ----------
        data : str
            The audio data in base64 encoded string format.
        """
        self.audio_status = AudioStatus.deserialize(data.payload.to_bytes())

        if self.audio_status.status_speaker == AudioStatus.STATUS_SPEAKER.ACTIVE.value:
            with self._lock:
                if not self._is_tts_active:
                    state = AudioStatus(
                        header=prepare_header(),
                        status_mic=self.audio_status.status_mic,
                        status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
                        sentence_to_speak=String(""),
                    )

                    if self.pub:
                        self.pub.put(state.serialize())

                    self._is_tts_active = True

        if self.audio_status.status_speaker == AudioStatus.STATUS_SPEAKER.READY.value:
            with self._lock:
                if self._is_tts_active:
                    state = AudioStatus(
                        header=prepare_header(),
                        status_mic=self.audio_status.status_mic,
                        status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
                        sentence_to_speak=String(""),
                    )

                    if self.pub:
                        self.pub.put(state.serialize())

                    self._is_tts_active = False

    def on_tts_state_change(self, is_active: bool):
        """
        Updates the TTS active state to control audio capture behavior.

        When TTS is active, audio capture is temporarily suspended to prevent
        capturing the TTS output.

        Parameters
        ----------
        is_active : bool
            True if TTS is currently playing, False otherwise
        """
        with self._lock:
            self._is_tts_active = is_active
            logger.info(f"TTS active state changed to: {is_active}")

    def register_audio_data_callback(self, audio_callback: Callable):
        """
        Registers a callback function for audio data processing.

        Parameters
        ----------
        callback : Callable
            Function to be called with audio data chunks
        """
        if audio_callback is None:
            logger.warning("Audio data callback is None, not registering")
            return

        if audio_callback not in self._audio_data_callbacks:
            self._audio_data_callbacks.append(audio_callback)
            logger.info("Registered new audio data callback")
            return

        logger.warning("Audio data callback already registered")
        return

    def start(self) -> "AudioRTSPInputStream":
        """
        Initializes and starts the audio capture stream.

        This method starts the RTSP audio processing thread.

        Returns
        -------
        AudioInputStream
            The current instance for method chaining

        Raises
        ------
        Exception
            If there are errors starting the RTSP stream
        """
        if not self.running:
            return self

        try:
            self._start_rtsp_audio_processor_thread()
            self._start_audio_callback_thread()

            logger.info(f"Started RTSP audio stream: {self._rtsp_url}")

        except Exception as e:
            logger.error(f"Error starting RTSP audio stream: {e}")
            raise

        return self

    def _start_rtsp_audio_processor_thread(self):
        """
        Starts the audio processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if (
            self._rtsp_audio_processor_thread is None
            or not self._rtsp_audio_processor_thread.is_alive()
        ):
            self._rtsp_audio_processor_thread = mp.Process(
                target=rtsp_audio_processor,
                args=(
                    self._rtsp_url,
                    self._rate,
                    self._chunk,
                    self._buff,
                    self._control_queue,
                    get_logging_config(),
                ),
            )
            self._rtsp_audio_processor_thread.start()
            logger.info("Started RTSP audio processing thread")

    def _start_audio_callback_thread(self):
        """
        Starts the audio callback processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if (
            self._audio_callback_thread is None
            or not self._audio_callback_thread.is_alive()
        ):
            self._audio_callback_thread = threading.Thread(
                target=self.on_audio, daemon=True
            )
            self._audio_callback_thread.start()
            logger.info("Started audio callback processing thread")

    def _fill_buffer(self, in_data: bytes):
        """
        Fill the audio buffer with processed RTSP data.

        This replaces the PyAudio stream callback mechanism.

        Parameters
        ----------
        in_data : bytes
            The captured audio data from RTSP stream
        """
        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)

    def fill_buffer_remote(self, data: str):
        """
        Callback function for remote audio data to fill the audio buffer.

        Parameters
        ----------
        data : str
        """
        try:
            audio_data = json.loads(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON data: {e}")
            return

        if "audio" not in audio_data:
            logger.error("Received remote audio data without 'audio' key")
            return

        in_data = base64.b64decode(audio_data["audio"])
        rate = audio_data.get("rate", self._rate)
        language_code = audio_data.get("language_code", self._language_code)

        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)
                self._rate = rate
                self._language_code = language_code

    def generator(self) -> Generator[Dict[str, Union[bytes, int]], None, None]:
        """
        Generates a stream of audio data chunks.

        This generator yields audio data chunks, combining multiple chunks when
        available to reduce processing overhead. It skips yielding data when
        TTS is active.

        Yields
        ------
        Dict[str, Union[bytes, int]]
            Dictionary containing base64 encoded audio, rate, and language_code
        """
        while self.running:
            chunk = self._buff.get()
            if chunk is None:
                return

            with self._lock:
                if self._is_tts_active:
                    continue

            # Collect additional chunks that are immediately available
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        assert self.running
                    if chunk:
                        data.append(chunk)
                except queue.Empty:
                    break

            response = {
                "audio": base64.b64encode(b"".join(data)).decode("utf-8"),
                "rate": self._rate,
                "language_code": self._language_code,
            }
            for audio_callback in self._audio_data_callbacks:
                audio_callback(json.dumps(response))

            yield response

    def on_audio(self):
        """Audio processing loop"""
        for _ in self.generator():
            if not self.running:
                break
        pass

    def stop(self):
        """
        Stops the audio stream and cleans up resources.

        This method stops the RTSP stream and ensures the audio processing
        thread is properly shut down.
        """
        self.running = False

        if self.session:
            self.session.close()

        # Clean up the audio processing thread
        if (
            self._rtsp_audio_processor_thread
            and self._rtsp_audio_processor_thread.is_alive()
        ):
            self._rtsp_audio_processor_thread.join(timeout=1.0)

        # Clean up the audio callback thread
        if self._audio_callback_thread and self._audio_callback_thread.is_alive():
            self._audio_callback_thread.join(timeout=1.0)

        self._buff.put(None)
        self._control_queue.put("STOP")
        logger.info("Stopped RTSP audio stream")
