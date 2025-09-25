from __future__ import annotations

import re
import time
import threading
import queue
from typing import Callable, Optional, Tuple, TYPE_CHECKING
import logging
import cv2

logging.basicConfig(level=logging.INFO)

if TYPE_CHECKING:
    import numpy as np


# ======================================================================
# Capture builders
# ======================================================================


def build_cam_capture(
    cam_index: int,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    """Open a V4L2 camera via OpenCV and set basic properties.

    Parameters
    ----------
    cam_index : int
        V4L2 index (e.g., 0 for /dev/video0).
    width, height : int
        Requested frame size.
    fps : int
        Requested frame rate.

    Returns
    -------
    cv2.VideoCapture
        Opened capture object (check `.isOpened()`).
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        # Fallback: explicit V4L2 backend
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

    # Reduce buffering to lower latency / avoid backpressure
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _device_index_from_path(device: str) -> int:
    """Extract numeric camera index from a device path like '/dev/video2'.

    Parameters
    ----------
    device : str
        Device path.

    Returns
    -------
    int
        Parsed index or 0 if not found.
    """
    m = re.search(r"(\d+)$", device)
    return int(m.group(1)) if m else 0


def build_gst_capture(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> Optional[cv2.VideoCapture]:
    """Open a UVC camera using a GStreamer pipeline. Tries MJPEG then YUY2.

    Pipelines are configured for low latency:
      - `io-mode=2` for v4l2src (DMABUF when available)
      - leaky queues and `appsink drop=true`
      - `videorate` and fixed caps to keep a steady format

    Parameters
    ----------
    device : str
        Video device path (e.g., '/dev/video0').
    width, height, fps : int
        Desired capture format.

    Returns
    -------
    Optional[cv2.VideoCapture]
        Opened GStreamer capture, or None if both attempts fail.
    """
    mjpeg = (
        f"v4l2src device={device} io-mode=2 "
        f"! image/jpeg,framerate={fps}/1,width={width},height={height} "
        f"! queue leaky=downstream max-size-buffers=1 "
        f"! jpegdec ! videoconvert ! videorate "
        f"! video/x-raw,format=BGR,framerate={fps}/1,width={width},height={height} "
        f"! appsink drop=true max-buffers=1 sync=false"
    )
    yuy2 = (
        f"v4l2src device={device} io-mode=2 "
        f"! video/x-raw,format=YUY2,framerate={fps}/1,width={width},height={height} "
        f"! queue leaky=downstream max-size-buffers=1 "
        f"! videoconvert ! videorate "
        f"! video/x-raw,format=BGR,framerate={fps}/1,width={width},height={height} "
        f"! appsink drop=true max-buffers=1 sync=false"
    )

    for pipe in (mjpeg, yuy2):
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            logging.info("[gst] using pipeline:", pipe)
            return cap

    logging.warning("[gst] both MJPEG and YUY2 pipelines failed for device=%s", device)
    # Final fallback: plain OpenCV V4L2
    idx = _device_index_from_path(device)
    return build_cam_capture(idx, width, height, fps)


# ======================================================================
# Writers
# ======================================================================


class AsyncVideoWriter:
    """Non-blocking wrapper over cv2.VideoWriter.

    Frames are queued and written on a background daemon thread.
    If the queue is full, frames are dropped instead of blocking the main loop.

    Parameters
    ----------
    writer : cv2.VideoWriter
        An already opened writer object (e.g., from `open_nvenc_rtmp_writer`).
    queue_size : int, optional
        Max buffered frames; small values keep latency low (default 1).
    """

    def __init__(self, writer: cv2.VideoWriter, queue_size: int = 1):
        self._w = writer
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def write(self, frame) -> bool:
        """Enqueue a frame for writing (drops when queue is full)."""
        if self._w is None:
            return False
        try:
            self._q.put_nowait(frame)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        try:
            while not self._stop.is_set() and self._w is not None:
                try:
                    f = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self._w.write(f)
                except Exception as e:
                    logging.warning(f"[warn] async writer error, stopping: {e}")
                    break
        finally:
            try:
                if self._w is not None:
                    self._w.release()
            except Exception:
                pass
            self._w = None

    def close(self) -> None:
        """Signal the writer thread to stop and release resources."""
        self._stop.set()
        if self._thr.is_alive():
            self._thr.join(timeout=1.0)

    def is_open(self) -> bool:
        """Return True if the underlying writer is still open."""
        return self._w is not None


# put near other imports
from urllib.parse import urlparse


def open_nvenc_rtmp_writer(
    rtmp_url: str, width: int, height: int, fps_in: float
) -> Optional[cv2.VideoWriter]:
    """
    Create a GStreamer RTMP/RTMPS writer.
    - Prefer rtmpsink (librtmp) for plain RTMP URLs (more tolerant of non /app/stream paths).
    - Try NVENC first, then CPU x264.
    - Use minimal, ffmpeg-like pipelines to avoid caps negotiation issues.
    """
    fps = int(round(fps_in if fps_in and fps_in > 0 else 25))
    gop = max(1, fps)

    u = urlparse(rtmp_url)
    scheme = (u.scheme or "rtmp").lower()
    port = u.port or (443 if scheme == "rtmps" else 1935)

    # Choose sink preference:
    # - for rtmp:// on 1935 (your cloud case), try rtmpsink first (librtmp),
    #   then rtmp2sink. For rtmps:// prefer rtmp2sink first.
    if scheme == "rtmps":
        sink_order = ["rtmp2sink", "rtmpsink"]
    else:
        sink_order = ["rtmpsink", "rtmp2sink"]

    # Common appsrc (live timestamps)
    common_src = (
        "appsrc is-live=true do-timestamp=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        "! queue leaky=downstream max-size-buffers=1 "
    )

    # NVENC (Jetson) minimal path
    nvenc = (
        "! videoconvert ! nvvidconv "
        "! video/x-raw(memory:NVMM),format=NV12 "
        f"! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=2500000 "
        f"iframeinterval={gop} idrinterval={gop} preset-level=1 "
        "! h264parse config-interval=1 "
        "! flvmux streamable=true "
    )

    # CPU x264 minimal path (mirrors your ffmpeg flags)
    x264 = (
        "! videoconvert "
        f"! x264enc tune=zerolatency speed-preset=ultrafast bitrate=2500 key-int-max={gop} bframes=0 byte-stream=true "
        "! h264parse config-interval=1 "
        "! flvmux streamable=true "
    )

    # Build candidate pipelines in the right order
    candidates = []
    for sink in sink_order:
        sink_str = f'! {sink} location="{rtmp_url}" sync=false async=false'
        candidates.append(common_src + nvenc + sink_str)
        candidates.append(common_src + x264 + sink_str)

    # Try each candidate until one opens
    for pipe in candidates:
        w = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        if w.isOpened():
            which_enc = "NVENC" if "nvv4l2h264enc" in pipe else "x264"
            which_sink = "rtmp2sink" if "rtmp2sink" in pipe else "rtmpsink"
            logging.info(
                f"[rtmp] using {which_enc} -> {which_sink} ({scheme}://:{port})"
            )
            return w

    return None


def build_file_writer(
    path: str,
    W: int,
    H: int,
    fps: float,
    use_nvenc: bool,
) -> Optional[cv2.VideoWriter]:
    """Create a local MP4 writer via NVENC (GStreamer) or CPU fallback.

    NVENC path writes H.264 into MP4 (fallback to MKV if MP4 muxer unavailable).
    CPU path uses OpenCV's simple `mp4v` fourcc.

    Parameters
    ----------
    path : str
        Output file path (e.g., '/tmp/out.mp4').
    W, H : int
        Frame size.
    fps : float
        Frames per second (rounded for pipeline caps).
    use_nvenc : bool
        Prefer Jetson NVENC when True.

    Returns
    -------
    Optional[cv2.VideoWriter]
        Opened writer or None.
    """
    fps_i = int(round(fps if fps > 0 else 25))

    if use_nvenc:
        pipe = (
            "appsrc is-live=true do-timestamp=true format=time "
            f"caps=video/x-raw,format=BGR,width={W},height={H},framerate={fps_i}/1 "
            "! queue leaky=downstream max-size-buffers=1 "
            "! videoconvert ! nvvidconv "
            "! video/x-raw(memory:NVMM),format=NV12 "
            "! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 "
            "! h264parse ! mp4mux "
            f"! filesink location={path} sync=false"
        )
        w = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, fps_i, (W, H))
        if w.isOpened():
            logging.info("[file] using NVENC MP4 pipeline:", path)
            return w

        # MKV fallback (sometimes MP4 muxer is missing)
        mkv_path = f"{path.rsplit('.', 1)[0]}.mkv"
        pipe_mkv = pipe.replace("mp4mux", "matroskamux").replace(path, mkv_path)
        w = cv2.VideoWriter(pipe_mkv, cv2.CAP_GSTREAMER, 0, fps_i, (W, H))
        if w.isOpened():
            logging.info("[file] using NVENC MKV pipeline:", mkv_path)
            return w

    # CPU fallback (OpenCV container)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w2 = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, (W, H))
    if w2.isOpened():
        logging.info("[file] using CPU mp4v:", path)
        return w2
    return None


# ======================================================================
# Robust helpers
# ======================================================================


def safe_read(
    cap: cv2.VideoCapture,
    max_retries: int = 5,
    sleep_sec: float = 0.02,
) -> Tuple[bool, Optional["np.ndarray"]]:
    """Read a frame with small retries to survive transient stalls.

    Parameters
    ----------
    cap : cv2.VideoCapture
        Opened capture.
    max_retries : int
        Number of quick retries before giving up.
    sleep_sec : float
        Sleep between retries.

    Returns
    -------
    (bool, ndarray | None)
        (ok, frame). If still failing, returns (False, None).
    """
    for _ in range(max_retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return True, frame
        time.sleep(sleep_sec)
    return False, None


def reopen_capture(
    cap: Optional[cv2.VideoCapture],
    builder: Callable[..., Optional[cv2.VideoCapture]],
    *args,
    **kwargs,
) -> Optional[cv2.VideoCapture]:
    """Release and rebuild a capture (e.g., when device briefly disappears).

    Parameters
    ----------
    cap : cv2.VideoCapture | None
        Existing capture to release (ignored if None).
    builder : Callable[..., cv2.VideoCapture | None]
        Function to create a new capture (e.g., `build_gst_capture`).
    *args, **kwargs :
        Passed through to the builder.

    Returns
    -------
    Optional[cv2.VideoCapture]
        Reopened capture or None if open failed.
    """
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    return builder(*args, **kwargs)
