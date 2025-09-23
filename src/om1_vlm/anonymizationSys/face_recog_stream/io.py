from __future__ import annotations

from typing import Optional

import cv2


def build_cam_capture(
    cam_index: int, width: int, height: int, fps: int
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
        Opened capture object (check .isOpened()).
    """
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def build_gst_capture(
    device: str, width: int, height: int, fps: int
) -> Optional[cv2.VideoCapture]:
    """Open a UVC camera using a GStreamer pipeline (MJPEG→BGR fallback to YUY2).

    Parameters
    ----------
    device : str
        Video device, e.g. "/dev/video0".
    width, height, fps : int
        Desired capture format.

    Returns
    -------
    Optional[cv2.VideoCapture]
        Opened capture or None if both attempts failed.
    """
    mjpeg = (
        f"v4l2src device={device} ! image/jpeg,framerate={fps}/1,width={width},height={height} "
        f"! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )
    yuy2 = (
        f"v4l2src device={device} ! video/x-raw,format=YUY2,framerate={fps}/1,width={width},height={height} "
        f"! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1"
    )
    for pipe in (mjpeg, yuy2):
        cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print("[gst] using pipeline:", pipe)
            return cap
    return None


def open_nvenc_rtmp_writer(
    rtmp_url: str, width: int, height: int, fps_in: float
) -> Optional[cv2.VideoWriter]:
    """Create a GStreamer RTMP writer using Jetson NVENC (fallback to x264).

    Parameters
    ----------
    rtmp_url : str
        Full RTMP URL (e.g., mediamtx or cloud endpoint).
    width, height : int
        Frame size.
    fps_in : float
        Input capture FPS; rounded to int for the pipeline.

    Returns
    -------
    Optional[cv2.VideoWriter]
        Opened writer or None if both NVENC and x264 pipelines failed.
    """
    fps = int(round(fps_in if fps_in and fps_in > 0 else 25))

    pipe1 = (
        "appsrc is-live=true do-timestamp=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        "! queue leaky=downstream max-size-buffers=1 "
        "! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
        "! nvv4l2h264enc insert-sps-pps=true idrinterval=30 iframeinterval=30 control-rate=1 bitrate=4000000 preset-level=1 "
        "! h264parse config-interval=1 ! flvmux streamable=true "
        f'! rtmpsink location="{rtmp_url}" sync=false'
    )
    w = cv2.VideoWriter(pipe1, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if w.isOpened():
        print("[rtmp] using NVENC rtmpsink")
        return w

    pipe2 = pipe1.replace("rtmpsink", "rtmp2sink")
    w = cv2.VideoWriter(pipe2, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if w.isOpened():
        print("[rtmp] using NVENC rtmp2sink")
        return w

    pipe3 = (
        "appsrc is-live=true do-timestamp=true format=time "
        f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 "
        "! queue leaky=downstream max-size-buffers=1 ! videoconvert "
        "! x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max=30 "
        "! video/x-h264,stream-format=byte-stream,alignment=au ! h264parse config-interval=1 "
        "! flvmux streamable=true "
        f'! rtmpsink location="{rtmp_url}" sync=false'
    )
    w = cv2.VideoWriter(pipe3, cv2.CAP_GSTREAMER, 0, fps, (width, height))
    if w.isOpened():
        print("[rtmp] fallback to CPU x264")
        return w
    return None


def build_rtmp_writer(
    rtmp_url: str, W: int, H: int, fps: int, use_nvenc: bool
) -> Optional[cv2.VideoWriter]:
    """Create an RTMP writer via NVENC or CPU x264.

    Parameters
    ----------
    rtmp_url : str
        Destination RTMP URL.
    W, H : int
        Frame size.
    fps : int
        Frames per second.
    use_nvenc : bool
        Prefer Jetson NVENC pipeline when True.

    Returns
    -------
    Optional[cv2.VideoWriter]
        Opened writer or None.
    """
    fps_i = int(round(fps if fps > 0 else 25))
    if use_nvenc:
        pipe = (
            f"appsrc caps=video/x-raw,format=BGR,width={W},height={H},framerate={fps_i}/1 "
            f"! queue leaky=downstream max-size-buffers=1 "
            f"! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 iframeinterval=30 "
            f'! h264parse ! flvmux streamable=true ! rtmpsink location="{rtmp_url}" sync=false'
        )
        w = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, fps_i, (W, H))
        if w.isOpened():
            print("[rtmp] using NVENC pipeline")
            return w
    # Fallback CPU
    pipe2 = (
        f"appsrc caps=video/x-raw,format=BGR,width={W},height={H},framerate={fps_i}/1 "
        f"! queue leaky=downstream max-size-buffers=1 ! videoconvert "
        f"! x264enc speed-preset=ultrafast tune=zerolatency bitrate=2000 key-int-max=30 ! video/x-h264,profile=baseline "
        f'! h264parse ! flvmux streamable=true ! rtmpsink location="{rtmp_url}" sync=false'
    )
    w2 = cv2.VideoWriter(pipe2, cv2.CAP_GSTREAMER, 0, fps_i, (W, H))
    if w2.isOpened():
        print("[rtmp] using x264 pipeline")
        return w2
    return None


def build_file_writer(
    path: str, W: int, H: int, fps: float, use_nvenc: bool
) -> Optional[cv2.VideoWriter]:
    """Create a local MP4 writer via NVENC or CPU fallback.

    Parameters
    ----------
    path : str
        Output file path (.mp4).
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
            f"appsrc caps=video/x-raw,format=BGR,width={W},height={H},framerate={fps_i}/1 "
            f"! queue leaky=downstream max-size-buffers=1 ! videoconvert "
            f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 "
            f"! h264parse ! mp4mux ! filesink location={path} sync=false"
        )
        w = cv2.VideoWriter(pipe, cv2.CAP_GSTREAMER, 0, fps_i, (W, H))
        if w.isOpened():
            print("[file] using NVENC mp4 pipeline:", path)
            return w
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w2 = cv2.VideoWriter(path, fourcc, fps if fps > 0 else 25.0, (W, H))
    return w2 if w2.isOpened() else None
