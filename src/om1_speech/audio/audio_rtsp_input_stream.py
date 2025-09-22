# Description: Audio stream class for capturing audio from RTSP streams
# Updated to use RTSP instead of microphone input while maintaining same API

import base64
import json
import logging
import queue
import threading
import time
from typing import Callable, Dict, Generator, List, Optional, Union

import av
import numpy as np

from zenoh_msgs import AudioStatus, String, open_zenoh_session, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


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
    remote_input : bool, optional
        If True, indicates that the audio input is from a remote source.
    """

    def __init__(
        self,
        rtsp_url: str = "rtsp://localhost:8554/live",
        rate: int = 16000,
        chunk: Optional[int] = None,
        audio_data_callback: Optional[Callable] = None,
        audio_data_callbacks: Optional[List[Callable]] = None,
        language_code: Optional[str] = None,
        remote_input: bool = False,
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
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()

        # Resampler
        self.resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=self._rate,
        )

        # RTSP container and stream
        self._rtsp_container: Optional[av.container.InputContainer] = None
        self._rtsp_audio_stream = None

        # Audio processing thread
        self._audio_thread: Optional[threading.Thread] = None

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

        self.remote_input = remote_input
        if self.remote_input:
            logger.info(
                "Remote input is enabled, skipping RTSP audio input initialization"
            )
            return

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
            if self.remote_input:
                logger.info(
                    "Remote input is enabled, skipping RTSP stream initialization"
                )
            else:
                logger.info("Starting RTSP audio capture")

            self._start_audio_thread()

            logger.info(f"Started RTSP audio stream: {self._rtsp_url}")

        except Exception as e:
            logger.error(f"Error starting RTSP audio stream: {e}")
            raise

        return self

    def _start_audio_thread(self):
        """
        Starts the audio processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread = threading.Thread(
                target=self._rtsp_audio_loop, daemon=True
            )
            self._audio_thread.start()
            logger.info("Started RTSP audio processing thread")

    def _rtsp_audio_loop(self):
        """
        Main RTSP audio processing loop with automatic reconnection.

        This replaces the PyAudio callback mechanism with RTSP frame processing.
        """
        reconnect_delay = 2.0

        while self.running:
            try:
                logger.info(f"Opening RTSP stream: {self._rtsp_url}")

                self._rtsp_container = av.open(
                    self._rtsp_url, options={"rtsp_transport": "tcp"}
                )

                if not self._rtsp_container.streams.audio:
                    logger.error(
                        f"No audio stream found in RTSP. Retrying in {reconnect_delay} seconds..."
                    )
                    time.sleep(reconnect_delay)
                    continue

                self._rtsp_audio_stream = self._rtsp_container.streams.audio[0]
                logger.info("RTSP audio stream connected successfully")

                audio_buffer = np.array([], dtype=np.int16)

                for frame in self._rtsp_container.decode(self._rtsp_audio_stream):
                    if not self.running:
                        break

                    frame_data = self._process_rtsp_frame(frame)
                    if frame_data:
                        audio_buffer = np.concatenate([audio_buffer, frame_data])

                        if len(audio_buffer) >= self._chunk:
                            chunk_data = audio_buffer[: self._chunk]
                            audio_buffer = audio_buffer[self._chunk :]

                            pcm = chunk_data.tobytes()
                            self._fill_buffer(pcm)

            except Exception as e:
                logger.error(f"RTSP audio error: {e}")
                if self._rtsp_container:
                    self._rtsp_container.close()
                    self._rtsp_container = None

                if self.running:
                    logger.info(f"Reconnecting in {reconnect_delay} seconds...")
                    time.sleep(reconnect_delay)

        if self._rtsp_container:
            self._rtsp_container.close()

    def _process_rtsp_frame(self, frame: av.AudioFrame) -> Optional[bytes]:
        """
        Process RTSP audio frame and convert to bytes.

        Parameters
        ----------
        frame : av.AudioFrame
            Audio frame from RTSP stream

        Returns
        -------
        Optional[bytes]
            Processed audio data in bytes, or None if processing fails
        """
        try:
            frame = self.resampler.resample(frame)[0]
            frame_data = frame.to_ndarray().astype(np.int16).flatten()
            return frame_data
        except Exception as e:
            logger.error(f"Error processing RTSP frame: {e}")
            return None

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

        # Close RTSP container
        if self._rtsp_container:
            self._rtsp_container.close()

        # Clean up the audio processing thread
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

        self._buff.put(None)
        logger.info("Stopped RTSP audio stream")
