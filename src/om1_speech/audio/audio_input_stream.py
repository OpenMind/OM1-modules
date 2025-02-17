# Description: Audio stream class for capturing audio from a microphone
# A partial of code comes from https://github.com/nvidia-riva/python-clients/blob/main/riva/client/audio_io.py

import logging
import queue
import threading
from typing import Any, Callable, Generator, Optional, Tuple, Union

import pyaudio

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioInputStream:
    """
    A class for capturing and managing real-time audio input from a microphone.

    This class provides functionality to capture audio data from a specified or default
    microphone device, process it in chunks, and make it available through a generator
    or callback mechanism. It supports Text-to-Speech (TTS) integration by temporarily
    suspending audio capture during TTS playback.

    Parameters
    ----------
    rate : int, optional
        The sampling rate in Hz for audio capture (default: 16000)
    chunk : int, optional
        The size of each audio chunk in frames (default: 4048)
    device : Optional[Union[str, int, float, Any]], optional
        The input device identifier. Can be device index or name. If None,
        uses system default input device (default: None)
    device_name: str, optional
        The input device name. If None, uses the first available input device.
        (default: None)
    audio_data_callback : Optional[Callable], optional
        A callback function that receives audio data chunks (default: None)
    """

    def __init__(
        self,
        rate: int = 16000,
        chunk: int = 4048,
        device: Optional[Union[str, int, float, Any]] = None,
        device_name: str = None,
        audio_data_callback: Optional[Callable] = None,
    ):
        self._rate = rate
        self._chunk = chunk
        self._device = device
        self._device_name = device_name

        # Callback for audio data
        self.audio_data_callback = audio_data_callback

        # Flag to indicate if TTS is active
        self._is_tts_active: bool = False

        # Thread-safe buffer for audio data
        self._buff: queue.Queue[Optional[bytes]] = queue.Queue()

        # audio interface and stream
        self._audio_interface: pyaudio.PyAudio = pyaudio.PyAudio()
        self._audio_stream: Optional[pyaudio.Stream] = None

        # Audio processing thread
        self._audio_thread: Optional[threading.Thread] = None

        # Lock for thread safety
        self._lock = threading.Lock()

        self.running: bool = True

        if self._device is not None and self._device_name is not None:
            raise ValueError("Only one of device or device_name can be specified")

        try:
            input_device = None
            device_count = self._audio_interface.get_device_count()
            logger.info(f"Found {device_count} audio devices")

            if self._device is not None:
                input_device = self._audio_interface.get_device_info_by_index(
                    self._device
                )
                logger.info(
                    f"Selected input device: {input_device['name']} ({self._device})"
                )
                if input_device["maxInputChannels"] == 0:
                    raise ValueError("Selected input device has no input channels")
            elif self._device_name is not None:
                available_devices = []
                for i in range(device_count):
                    device_info = self._audio_interface.get_device_info_by_index(i)
                    if device_info["maxInputChannels"] > 0:
                        device_name = device_info["name"]
                        available_devices.append({"name": device_name, "index": i})
                        if self._device_name.lower() in device_name.lower():
                            input_device = device_info
                            self._device = i
                            break
                if input_device is None:
                    raise ValueError(
                        f"Input device '{self._device_name}' not found. Available devices: {available_devices}"
                    )
            else:
                input_device = self._audio_interface.get_default_input_device_info()
                self._device = input_device["index"]
                logger.info(
                    f"Default input device: {input_device['name']} ({self._device})"
                )

            if input_device is None:
                raise ValueError("No input device found")

            logger.info(
                f"Selected input device: {input_device['name']} ({self._device})"
            )

        except Exception as e:
            logger.error(f"Error initializing audio input: {e}")
            self._audio_interface.terminate()
            raise

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

    def start(self) -> "AudioInputStream":
        """
        Initializes and starts the audio capture stream.

        This method sets up the PyAudio interface, configures the input device,
        and starts the audio processing thread.

        Returns
        -------
        AudioInputStream
            The current instance for method chaining

        Raises
        ------
        Exception
            If there are errors initializing the audio interface or opening the stream
        """
        if not self.running:
            return self

        try:
            self._audio_stream = self._audio_interface.open(
                format=pyaudio.paInt16,
                input_device_index=self._device,
                channels=1,
                rate=self._rate,
                input=True,
                frames_per_buffer=self._chunk,
                stream_callback=self._fill_buffer,
            )

            # Start the audio processing thread
            self._start_audio_thread()

            logger.info(f"Started audio stream with device {self._device}")

        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            self._audio_interface.terminate()
            raise

        return self

    def _start_audio_thread(self):
        """
        Starts the audio processing thread if it's not already running.

        The thread runs as a daemon to ensure it terminates when the main program exits.
        """
        if self._audio_thread is None or not self._audio_thread.is_alive():
            self._audio_thread = threading.Thread(target=self.on_audio, daemon=True)
            self._audio_thread.start()
            logger.info("Started audio processing thread")

    def _fill_buffer(
        self, in_data: bytes, frame_count: int, time_info: dict, status_flags: int
    ) -> Tuple[None, int]:
        """
        Callback function for the PyAudio stream to fill the audio buffer.

        This method is called by PyAudio when new audio data is available.
        It adds the data to the buffer queue if TTS is not active.

        Parameters
        ----------
        in_data : bytes
            The captured audio data
        frame_count : int
            Number of frames in the audio data
        time_info : dict
            Timing information from PyAudio
        status_flags : int
            Status flags from PyAudio

        Returns
        -------
        Tuple[None, int]
            A tuple containing None and pyaudio.paContinue
        """
        with self._lock:
            if not self._is_tts_active:
                self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self) -> Generator[bytes, None, None]:
        """
        Generates a stream of audio data chunks.

        This generator yields audio data chunks, combining multiple chunks when
        available to reduce processing overhead. It skips yielding data when
        TTS is active.

        Yields
        ------
        bytes
            Combined audio data chunks
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

            if self.audio_data_callback:
                self.audio_data_callback(b"".join(data))

            yield b"".join(data)

    def on_audio(self):
        """Audio processing loop"""
        for _ in self.generator():
            if not self.running:
                break
        pass

    def stop(self):
        """
        Stops the audio stream and cleans up resources.

        This method stops the audio stream, terminates the PyAudio interface,
        and ensures the audio processing thread is properly shut down.
        """
        self.running = False

        if self._audio_stream:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._audio_interface:
            self._audio_interface.terminate()

        # Clean up the audio processing thread
        if self._audio_thread and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=1.0)

        self._buff.put(None)
        logger.info("Stopped audio stream")
