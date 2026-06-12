import base64
import json
import logging
import struct
from queue import Empty, Queue
from typing import Any, Dict, Optional

from ..interfaces import AudioStreamInputInterface

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class AudioStreamInput(AudioStreamInputInterface):
    """
    A class for managing audio input streaming from WebSocket connections.

    This class provides a queue-based buffer for handling incoming audio data
    from WebSocket connections and makes it available for processing through
    a simple interface.

    Parameters
    ----------
    None
    """

    def __init__(self):
        self.running: bool = True
        self.audio_queue: Queue[Optional[Dict[str, Any]]] = Queue()

    def handle_ws_incoming_message(self, connection_id: str, message: Any):
        """
        Process incoming WebSocket messages containing audio data.

        Parameters
        ----------
        connection_id : str
            Identifier for the WebSocket connection
        message : Any
            The message received from the WebSocket connection,
            expected to be binary audio data
        """
        try:
            if isinstance(message, bytes):
                audio, rate = self._parse_binary(message)
                self.audio_queue.put({"audio": audio, "rate": rate})
            if isinstance(message, str):
                try:
                    message = json.loads(message)
                except json.JSONDecodeError:
                    logger.error("Error decoding JSON message")
                    return

                if "audio" not in message:
                    logger.error("Audio not found in message")
                    return
                audio = message["audio"]

                rate = 16000
                if "rate" in message:
                    rate = message["rate"]

                self.audio_queue.put({"audio": audio, "rate": rate})
            return
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def setup_audio_stream(self):
        """
        Set up the audio stream (placeholder method).

        Returns
        -------
        AudioStreamInput
            The current instance for method chaining
        """
        return self

    def get_audio_chunk(self) -> Optional[Dict[str, Any]]:  # type: ignore
        """
        Retrieve the next chunk of audio data from the queue.

        Returns
        -------
        Optional[Dict[str, Any]]
            A dictionary containing audio data and rate, or None if no data is available
        """
        try:
            data = self.audio_queue.get_nowait()
            return data
        except Empty:
            return None

    def stop(self):
        """
        Stop the audio stream processing.

        Sets the running flag to False to stop processing.
        """
        self.running = False

    @staticmethod
    def _parse_binary(data: bytes):
        """Parse binary format of ASR.

        Falls back to raw PCM at 16 kHz if the header is invalid.
        """
        if len(data) > 4:
            header_len = struct.unpack(">I", data[:4])[0]
            if 4 + header_len < len(data):
                try:
                    header = json.loads(data[4 : 4 + header_len])
                    pcm = data[4 + header_len :]
                    rate = header.get("rate", 16000)
                    return base64.b64encode(pcm).decode("utf-8"), rate
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        return base64.b64encode(data).decode("utf-8"), 16000
