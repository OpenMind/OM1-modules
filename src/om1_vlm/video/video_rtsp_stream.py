import asyncio
import base64
import inspect
import json
import logging
import threading
import time
from typing import Callable, List, Optional, Tuple

import cv2

root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)


class VideoRTSPStream:
    """
    Manages video capture and streaming from a RTSP stream.

    Provides functionality to capture video frames from a RTSP stream,
    process them, and stream them through a callback function.

    Parameters
    ----------
    rtsp_url : str
        The RTSP URL of the video stream.
    decode_format : str, optional
        The format to decode the RTSP stream, by default "H264"
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

    # Per-URL singleton registry
    _instances: dict[str, "VideoRTSPStream"] = {}
    _instances_lock = threading.Lock()

    def __new__(
        cls,
        rtsp_url: str = "rtsp://localhost:8554/live",
        decode_format: str = "H264",
        *args,
        **kwargs,
    ):
        with cls._instances_lock:
            existing = cls._instances.get(rtsp_url)
            if existing is not None:
                # already have a stream for this URL → reuse and bump refcount
                existing._refcount += 1
                logger.info(
                    f"Reusing existing VideoRTSPStream for {rtsp_url}, "
                    f"refcount={existing._refcount}"
                )
                return existing

            # first time this rtsp_url is seen → create new instance
            instance = super().__new__(cls)
            cls._instances[rtsp_url] = instance
            instance._refcount = 1
            logger.info(f"Created new VideoRTSPStream for {rtsp_url}, refcount=1")
            return instance

    def __init__(
        self,
        rtsp_url: str = "rtsp://localhost:8554/live",
        decode_format: str = "H264",
        frame_callback: Optional[Callable[[str], None]] = None,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
        fps: Optional[int] = 30,
        resolution: Optional[Tuple[int, int]] = (480, 640),
        jpeg_quality: int = 70,
    ):
        # Prevent reinitialising on subsequent "constructions" for same URL
        if getattr(self, "_initialized", False):
            # register any new callbacks on the existing instance
            try:
                if frame_callback is not None:
                    self.register_frame_callback(frame_callback)
                    logger.info(
                        f"VideoRTSPStream for {self.rtsp_url}: registered extra frame callback on reused stream"
                    )
                    logger.info(
                        f"Current number of callbacks: {len(self.frame_callbacks)}"
                    )
            except Exception as e:
                logger.error(
                    f"VideoRTSPStream for {self.rtsp_url}: failed to register extra callback(s) on reused stream: {e}"
                )

            # mismatch logging logic (decode_format only)
            if decode_format != self.decode_format:
                logger.info(
                    f"VideoRTSPStream for {self.rtsp_url} already initialized "
                    f"with decode_format={self.decode_format}, fps={self.fps}, "
                    f"resolution={self.resolution}, jpeg_quality={self.jpeg_quality}, "
                    f"ignoring new request with decode_format={decode_format}, "
                    f"fps={fps}, resolution={resolution}, jpeg_quality={jpeg_quality}"
                )
            return

        self._initialized = True

        # RTSP stream parameters
        self.rtsp_url = rtsp_url
        self.decode_format = decode_format

        self._video_thread: Optional[threading.Thread] = None

        # Callbacks for video frame data
        self.frame_callbacks = frame_callbacks or []
        self.register_frame_callback(frame_callback)

        # Video capture device
        self._cap = None

        self.running: bool = True

        self.fps = fps
        self.frame_delay = 1.0 / fps  # Calculate delay between frames
        self.resolution = resolution
        self.encode_quality = [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
        ]

        # Create a dedicated event loop for async tasks
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

    def _start_loop(self):
        """
        Set and run the event loop forever in a dedicated thread.
        """
        asyncio.set_event_loop(self.loop)
        logger.debug("Starting background event loop for video streaming.")
        self.loop.run_forever()

    def _release_capture(self):
        """
        Safely release the video capture device.
        """
        if self._cap is not None:
            try:
                self._cap.release()
                logger.debug("Released video capture device")
            except Exception as e:
                logger.warning(f"Error releasing capture: {e}")
            finally:
                self._cap = None

    def on_video(self):
        """
        Main video capture and processing loop.

        Captures frames from the RTMP stream, encodes them to base64,
        and sends them through the callback if registered.

        Raises
        ------
        Exception
            If video streaming encounters an error
        """
        while self.running:
            try:
                self._release_capture()

                self._cap = cv2.VideoCapture(self.rtsp_url)

                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap.set(cv2.CAP_PROP_FPS, self.fps)
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self._cap.set(
                    cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.decode_format)
                )

                if not self._cap.isOpened():
                    logger.error(
                        f"Error: Could not open RTSP stream at {self.rtsp_url}"
                    )
                    time.sleep(2)
                    continue

                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (
                    actual_width != self.resolution[0]
                    or actual_height != self.resolution[1]
                ):
                    logger.warning(
                        f"Camera doesn't support resolution {self.resolution}. Using {(actual_width, actual_height)} instead."
                    )
                    self.resolution = (actual_width, actual_height)

                frame_time = 1.0 / self.fps
                last_frame_time = time.perf_counter()

                while self.running:
                    current_time = time.perf_counter()
                    elapsed = current_time - last_frame_time

                    ret, frame = self._cap.read()

                    if not ret:
                        logger.error("Error reading frame from video stream")
                        time.sleep(0.1)
                        break

                    if elapsed <= 1.5 * frame_time and self.frame_callbacks:
                        _, buffer = cv2.imencode(".jpg", frame, self.encode_quality)
                        frame_base64 = base64.b64encode(buffer).decode("utf-8")

                        frame_data = json.dumps(
                            {"timestamp": time.time(), "frame": frame_base64}
                        )

                        for frame_callback in self.frame_callbacks:
                            if inspect.iscoroutinefunction(frame_callback):
                                asyncio.run_coroutine_threadsafe(
                                    frame_callback(frame_data), self.loop
                                )
                            else:
                                frame_callback(frame_data)

                    elapsed_time = time.perf_counter() - last_frame_time
                    if elapsed_time < frame_time:
                        time.sleep(frame_time - elapsed_time)
                    last_frame_time = time.perf_counter()

            except Exception as e:
                logger.error(f"Error streaming video: {e}")
            finally:
                self._release_capture()

            if self.running:
                time.sleep(2)

        logger.info("RTSP Video processing thread stopped")

    def _start_video_thread(self):
        """
        Initialize and start the video processing thread.

        Creates a new daemon thread for video processing if one isn't
        already running.
        """
        if self._video_thread is None or not self._video_thread.is_alive():
            self._video_thread = threading.Thread(target=self.on_video, daemon=True)
            self._video_thread.start()
            logger.info("Started video processing thread")

    def register_frame_callback(self, frame_callback: Callable[[str], None]):
        """
        Register a callback function for processed frames.

        Parameters
        ----------
        frame_callback : Callable[[str], None]
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
        self._start_video_thread()

    def stop(self):
        """
        Stop video capture and clean up resources.

        Decrement reference count and only actually stop the stream
        when the last user releases it.
        """
        with self._instances_lock:
            self._refcount -= 1
            logger.info(
                f"VideoRTSPStream.stop called for {self.rtsp_url}, "
                f"new refcount={self._refcount}"
            )

            if self._refcount > 0:
                # Still in use by other providers — do NOT tear it down.
                return

            # Last user: really stop and remove from registry
            self.running = False
            self._release_capture()

            if self._video_thread and self._video_thread.is_alive():
                self._video_thread.join(timeout=1.0)

            logger.info("Stopped video processing thread")

            # Remove from class-level registry
            try:
                del self._instances[self.rtsp_url]
            except KeyError:
                pass
