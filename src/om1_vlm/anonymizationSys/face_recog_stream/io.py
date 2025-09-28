from __future__ import annotations

import logging
import subprocess
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional, Tuple
from urllib.parse import urlparse

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


def open_capture(
    device: str,
    width: int,
    height: int,
    fps: int,
) -> Optional[cv2.VideoCapture]:
    """Open a UVC camera using a V4L2 pipeline. Tries MJPEG then YUY2.
    Parameters
    ----------
    device : str
        Video device path (e.g., '/dev/video0').
    width, height, fps : int
        Desired capture format.

    Returns
    -------
    Optional[cv2.VideoCapture]
        Opened V4L2 capture, or None if the camera can't be opened.
    """
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if cap.isOpened():
        return cap

    return None

# ======================================================================
# Writers
# ======================================================================

def open_ffmpeg_rtmp_writer(rtmp_url: str, width: int, height: int, fps: int):
    fourcc = cv2.VideoWriter_fourcc(*"H264")
    out = cv2.VideoWriter(
        rtmp_url,
        cv2.CAP_FFMPEG,
        fourcc,
        fps,
        (width, height),
    )
    if not out.isOpened():
        raise RuntimeError("Could not open FFmpeg RTMP writer")
    return out

def open_ffmpeg_rtsp_writer(rtsp_url: str, width: int, height: int, fps: int):
    # return None
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "h264",          # or "h264_nvenc" if you have NVIDIA GPU
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",   # use TCP to avoid UDP packet loss
        rtsp_url,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


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
