import asyncio
import base64
import inspect
import json
import logging
import time
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import zenoh

from zenoh_msgs import Image, open_zenoh_session

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VideoZenohStream:
    """
    Manages video capture and streaming from a zenoh topic.

    Provides functionality to capture video frames from a zenoh topic,
    process them, and stream them through a callback function.

    Parameters
    ----------
    topic : str
        The Zenoh topic to publish the video stream to. Defaults to "rgb_image".
    decode_format : str, optional
        The format to decode the zenoh stream, by default "H264"
    frame_callback : Optional[Callable[[str], None]], optional
    frame_callbacks : Optional[List[Callable[[str], None]]], optional
        List of callback functions to be called with base64 encoded frame data,
        by default None
    fps : Optional[int], optional
        Frames per second to capture.
        By default 30
    resolution : Optional[Tuple[int, int]], optional
        Resolution of the captured video frames.
        By default (480, 640)
    jpeg_quality : int, optional
        JPEG quality for encoding frames, by default 70
    """

    def __init__(
        self,
        topic: str = "rgb_image",
        decode_format: str = "H264",
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        resolution: Optional[Tuple[int, int]] = (480, 640),
        jpeg_quality: int = 70,
    ):
        # Zenoh video stream parameters
        self.topic = topic
        self.decode_format = decode_format

        # Callbacks for video frame data
        self.frame_callbacks = frame_callbacks or []
        self.register_frame_callback(frame_callback)

        self.running: bool = True

        self.resolution = resolution
        self.encode_quality = [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
        ]

        # Create a zenoh session
        self.zenoh_session = None
        try:
            self.zenoh_session = open_zenoh_session()
            self.zenoh_session.declare_subscriber(self.topic, self.zenoh_video_message)
            logger.info("Zenoh session opened successfully for video streaming")
        except Exception as e:
            logger.error(f"Failed to open Zenoh session: {e}")

        # Create a dedicated event loop for async tasks
        self.loop = asyncio.new_event_loop()

    def zenoh_video_message(self, sample: zenoh.Sample):
        """
        Callback function for Zenoh video messages.

        Parameters
        ----------
        sample : zenoh.Sample
            The video data in bytes.
        """
        if not self.running:
            return

        try:
            image_data = Image.deserialize(sample.payload.to_bytes())
            image_array = np.frombuffer(image_data.data, dtype=np.uint8)  # type: ignore

            if image_data.encoding == "rgb8":
                frame = image_array.reshape((image_data.height, image_data.width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif image_data.encoding == "bgr8":
                frame = image_array.reshape((image_data.height, image_data.width, 3))
            else:
                logger.warning(f"Unsupported encoding: {image_data.encoding}")
                return

            _, buffer = cv2.imencode(".jpg", frame, self.encode_quality)
            frame_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            frame_data = json.dumps({"timestamp": time.time(), "frame": frame_base64})

            for frame_callback in self.frame_callbacks:
                if inspect.iscoroutinefunction(frame_callback):
                    asyncio.run_coroutine_threadsafe(
                        frame_callback(frame_data), self.loop
                    )
                else:
                    frame_callback(frame_data)

        except Exception as e:
            logger.error(f"Error processing zenoh video message: {e}")

    def register_frame_callback(self, frame_callback: Optional[Callable[[str], None]]):
        """
        Register a callback function for processed frames.

        Parameters
        ----------
        frame_callback : Optional[Callable[[str], None]]
            Function to be called with base64 encoded frame data
        """
        if frame_callback is None:
            logger.warning("Frame callback is None, not registering")
            return

        if frame_callback not in self.frame_callbacks:
            self.frame_callbacks.append(frame_callback)
            logger.info("Registered new frame callback")
            return

        logger.warning("Frame callback already registered")
        return

    def start(self):
        """
        Start video capture and processing.

        Initializes the video processing thread and begins
        capturing frames.
        """
        self.running = True

    def stop(self):
        """
        Stop video capture and clean up resources.

        Stops the video processing loop and waits for the
        processing thread to finish.
        """
        self.running = False

        if self.zenoh_session:
            self.zenoh_session.close()
            logger.info("Closed Zenoh session for video streaming")

        logger.info("Stopped video processing thread")
