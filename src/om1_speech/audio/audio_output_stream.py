import argparse
import base64
import json
import logging
import shutil
import subprocess
import threading
import time
from queue import Queue
from typing import Callable, Dict, Optional

import requests
import zenoh

from zenoh_idl.status_msgs import AudioStatus
from zenoh_idl.std_msgs import String, prepare_header

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioOutputStream:
    """
    A class for managing audio output and text-to-speech (TTS) conversion.

    Parameters
    ----------
    url : str
        The URL endpoint for the text-to-speech service
    rate : int, optional
        The sampling rate in Hz for audio output (default: 8000)
    tts_state_callback : Optional[Callable], optional
        A callback function to receive TTS state changes (active/inactive)
        (default: None)
    headers : Optional[Dict[str, str]], optional
        Additional headers to include in the HTTP request (default: None)
    """

    def __init__(
        self,
        url: str,
        rate: int = 8000,
        tts_state_callback: Optional[Callable] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self._url = url
        self._rate = rate

        # Process headers
        self._headers = headers or {}
        if "Content-Type" not in self._headers:
            self._headers["Content-Type"] = "application/json"

        # Callback for TTS state
        self._tts_state_callback = tts_state_callback

        # Zenoh
        self.topic = "robot/status/audio"
        self.session = None
        self.pub = None
        self.audio_status = None

        try:
            self.session = zenoh.open(zenoh.Config())
            self.pub = self.session.declare_publisher(self.topic)
            self.session.declare_subscriber(self.topic, self.zenoh_audio_message)
        except Exception as e:
            logger.error(f"Failed to declare Zenoh subscriber: {e}")
            self.session = None

        # Pending requests queue
        self._pending_requests: Queue[Optional[str]] = Queue()

        # Slience audio for Bluetooth optimization
        self._silence_audio = self._create_silence_audio(100)
        self._silence_prefix = self._create_silence_audio(500)

        # Running state and last audio time
        self.running: bool = True
        self._last_audio_time = time.time()
        
        self._is_playing = False

        self._keep_alive_sound = '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'

    def zenoh_audio_message(self, data: str):
        """
        Callback function for Zenoh audio status messages.

        Parameters
        ----------
        data : str
            The audio data in base64 encoded string format.
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
        and plays them through the audio device.
        """
        while self.running:
            try:
                tts_request = self._pending_requests.get()
                response = requests.post(
                    self._url,
                    data=json.dumps(tts_request),
                    headers=self._headers,
                    timeout=(5, 15),
                )
                logger.info(f"Received TTS response: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    if "response" in result:
                        audio_data = result["response"]
                        self._write_audio(audio_data)
                        logger.debug(f"Processed TTS request: {tts_request}")
                else:
                    logger.error(
                        f"TTS request failed with status code {response.status_code}: {response.text}. Request details: {tts_request}"
                    )
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                continue

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
        return base64.b64encode(silence_bytes)

    def _keepalive_worker(self):
        """
        Background thread to play keepalive sounds every 60 seconds.
        """
        while self.running:
            current_time = time.time()
            if current_time - self._last_audio_time >= 10 and not self._is_playing:
                
                self._write_audio_bytes(self._keep_alive_sound, is_keepalive=True)
                # self._write_audio_bytes(self._silence_audio, is_keepalive=True)
                self._last_audio_time = current_time

            time.sleep(2)

    def _write_audio(self, audio_data: bytes):
        """
        Write audio data to the output stream with Bluetooth optimization.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        """
        self._last_audio_time = time.time()

        audio_bytes = self._silence_prefix + base64.b64decode(audio_data)

        self._write_audio_bytes(base64.b64encode(audio_bytes))

    def _write_audio_bytes(self, audio_data: bytes, is_keepalive: bool = False):
        """
        Write audio data to the output stream using ffplay.

        Parameters
        ----------
        audio_data : bytes
            The audio data to be written to the output stream
        is_keepalive : bool
            Whether this is a keepalive sound (suppresses callbacks)
        """
        audio_bytes = base64.b64decode(audio_data)

        if not is_installed("ffplay"):
            message = (
                "ffplay from ffmpeg not found, necessary to play audio. "
                "On mac you can install it with 'brew install ffmpeg'. "
                "On linux and windows you can install it from https://ffmpeg.org/"
            )
            logger.error(message)
            return

        if not is_keepalive:
            self._tts_callback(True)
            self._is_playing = True

            state = AudioStatus(
                header=prepare_header(),
                status_mic=(
                    self.audio_status.status_mic
                    if self.audio_status
                    else AudioStatus.STATUS_MIC.UNKNOWN.value
                ),
                status_speaker=AudioStatus.STATUS_SPEAKER.ACTIVE.value,
                sentence_to_speak=String(""),
                sentence_counter=(
                    self.audio_status.sentence_counter + 1 if self.audio_status else 0
                ),
            )

            if self.pub:
                self.pub.put(state.serialize())

        args = [
            "ffplay",
            "-autoexit",
            "-",
            "-nodisp",
            # It is not good at supporting audio_device_index from pyaudio
            # Reading the list from ffplay doesn't work either
            # "-audio_device_index",
            # str(self._device),
        ]

        if not is_keepalive:
            logging.info("Starting to play audio")

        proc = subprocess.Popen(
            args=args,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        proc.communicate(input=audio_bytes)
        exit_code = proc.poll()

        if exit_code != 0 and not is_keepalive:
            logger.error(f"Error playing audio: {exit_code}")

        if not is_keepalive:
            self._delayed_tts_false_callback()
            self._is_playing = False

            state = AudioStatus(
                header=prepare_header(),
                status_mic=(
                    self.audio_status.status_mic
                    if self.audio_status
                    else AudioStatus.STATUS_MIC.UNKNOWN.value
                ),
                status_speaker=AudioStatus.STATUS_SPEAKER.READY.value,
                sentence_to_speak=String(""),
                sentence_counter=(
                    self.audio_status.sentence_counter + 1 if self.audio_status else 0
                ),
            )

            if self.pub:
                self.pub.put(state.serialize())

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

    def _delayed_tts_false_callback(self, delay_seconds: float = 1.0):
        """
        Call the TTS callback after a delay without blocking the main thread.
        
        Parameters
        ----------
        delay_seconds : float
            The delay in seconds before calling the callback
        """
        def delayed_callback():
            time.sleep(delay_seconds)
            self._tts_callback(False)
        
        # Start the delayed callback in a separate thread
        thread = threading.Thread(target=delayed_callback, daemon=True)
        thread.start()

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

        if self.session:
            self.session.close()


def is_installed(lib_name: str) -> bool:
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
    args = parser.parse_args()

    audio_output = AudioOutputStream(args.tts_url, device=args.device, rate=args.rate)
    audio_output.start()
    audio_output.run_interactive()
    audio_output.stop()


if __name__ == "__main__":
    main()
