# -*- coding: utf-8 -*-
# om1_vlm/video/video_stream_blur_face.py

"""
Video capture + optional face anonymization (pixelation) pipeline.

This module launches a two-process pipeline:
- Capture process (V4L2 on Linux) that reads frames and pushes them into a queue
- Anonymization process (optional, TensorRT SCRFD) that pixelates faces
A main-thread drain loop encodes frames to JPEG (base64) and dispatches to callbacks.
"""

import asyncio
import base64
import inspect
import logging
import multiprocessing as mp
import platform
import threading
import time
from queue import Empty, Full
from typing import Callable, Iterable, List, Optional, Tuple

import cv2

from ..anonymizationSys.scrfd_trt_pixelate import (
    CONF_THRES,
    MAX_DETS,
    TOPK_PER_LEVEL,
    TRTInfer,
    apply_pixelation,
    draw_dets,
)
from .logging_info import get_logging_config, setup_logging_mp_child
from .video_utils import enumerate_video_devices

# Keep logger naming consistent with your other modules
root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

SENTINEL = ("__STOP__", None)


def _pick_camera() -> str:
    """
    Pick a camera device in a cross-platform manner (macOS/Linux),
    mirroring the style from your reference VideoStream implementation.

    Returns
    -------
    str
        Camera index (macOS) or device path (Linux).
    """
    devices = enumerate_video_devices()
    if platform.system() == "Darwin":
        camindex = 0 if devices else 0
    else:
        camindex = f"/dev/video{devices[0][0]}" if devices else "/dev/video0"
    logger.info(f"Using camera: {camindex}")
    return camindex


def proc_capture(
    out_q: mp.Queue,
    cam: str,
    res: Tuple[int, int],
    fps: int,
    buffer_frames: int,
    log_queue: Optional[mp.Queue] = None,
) -> None:
    """
    Capture frames from a camera in a dedicated process and push them to a queue.

    Parameters
    ----------
    out_q : mp.Queue
        Output queue for (timestamp, frame) tuples.
    cam : str
        Camera index/path, e.g. 0 (macOS) or '/dev/video0' (Linux).
    res : Tuple[int, int]
        Requested (width, height).
    fps : int
        Requested FPS.
    buffer_frames : int
        Desired driver-side capture buffer size (best effort).
    log_queue : Optional[mp.Queue], optional
        If provided, child process logs are routed to the main process.
    """
    if log_queue is not None:
        setup_logging_mp_child(log_queue, level=get_logging_config().log_level)

    log = logging.getLogger(root_package_name)
    cap = None

    try:
        # Prefer V4L2 on Linux; macOS ignores this backend id.
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            log.warning(f"[cap] failed to open {cam}, trying common fallbacks.")

        if not cap or not cap.isOpened():
            log.error(f"[cap] cannot open camera {cam}")
            return

        # Apply settings (best-effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        try:
            # MJPG can reduce CPU usage
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_frames)))
        except Exception:
            pass

        # Warn if camera doesn't honor the requested resolution
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (res[0], res[1]):
            log.warning(
                f"[cap] Camera doesn't support resolution {res}. Using {(actual_w, actual_h)} instead."
            )

        frame_time = 1.0 / max(1, fps)
        last = time.perf_counter()

        log.info(f"[cap] capture loop starting at ~{fps} FPS, device={cam}")
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("[cap] error reading frame; retrying shortly.")
                time.sleep(0.02)
                continue

            ts = time.time()
            pkt = (ts, frame)

            # Non-blocking put with drop policy
            try:
                out_q.put_nowait(pkt)
            except Full:
                try:
                    out_q.get_nowait()  # drop one stale
                except Empty:
                    pass
                try:
                    out_q.put_nowait(pkt)
                except Full:
                    pass

            elapsed = time.perf_counter() - last
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last = time.perf_counter()

    except KeyboardInterrupt:
        log.info("[cap] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        log.exception(f"[cap] unexpected error: {e}")
    finally:
        try:
            if cap is not None:
                cap.release()
                log.info("[cap] released video capture device")
        except Exception:
            pass
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        log.info("[cap] exit.")


def proc_anonymize(
    in_q: mp.Queue,
    out_q: mp.Queue,
    scrfd_cfg: dict,
    blur_enabled: bool,
    draw_boxes: bool,
    log_queue: Optional[mp.Queue] = None,
) -> None:
    """
    Optional anonymizer process: applies face detection & pixelation, then forwards frames.

    Parameters
    ----------
    in_q : mp.Queue
        Input queue receiving (timestamp, frame) tuples from capture process.
    out_q : mp.Queue
        Output queue to send processed (timestamp, frame) tuples to main process.
    scrfd_cfg : dict
        Configuration for the TensorRT SCRFD inferencer and pixelation parameters.
    blur_enabled : bool
        If True and engine_path is provided, apply anonymization.
    draw_boxes : bool
        If True, draw detection boxes for debug.
    log_queue : Optional[mp.Queue], optional
        If provided, child process logs are routed to the main process.
    """
    if log_queue is not None:
        setup_logging_mp_child(log_queue, level=get_logging_config().log_level)
    log = logging.getLogger(root_package_name)

    anonymizer = None
    cuda_ctx = None
    # sum_gpu_ms = 0.0
    # sum_pix_ms = 0.0
    # n_frames_anon = 0

    try:
        if blur_enabled and scrfd_cfg.get("engine_path"):
            import os

            import pycuda.driver as cuda

            cuda.init()
            dev_id = int(os.environ.get("OM1_CUDA_DEVICE", "0"))
            cuda_ctx = cuda.Device(dev_id).make_context()
            log.info(f"[anon] CUDA context created on device {dev_id}")

            try:
                anonymizer = _build_anonymizer(scrfd_cfg)
                log.info("[anon] anonymizer initialized.")
            except Exception:
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass
                log.exception("[anon] failed to initialize anonymizer.")
                raise

        log.info("[anon] anonymize loop starting (enabled=%s).", bool(anonymizer))
        while True:
            try:
                ts, frame = in_q.get(timeout=1.0)
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                log.info("[anon] received sentinel; exiting loop.")
                break

            if blur_enabled and anonymizer is not None:
                # t0 = time.perf_counter() #For logging info
                frame, dets, gpu_ms = anonymizer(frame)
                # For logging info
                # t1 = time.perf_counter()
                # pix_ms = (t1 - t0) * 1000.0 - (gpu_ms or 0.0)
                # if pix_ms < 0:
                #     pix_ms = 0.0

                if draw_boxes and dets is not None:
                    draw_dets(frame, dets)

                # For logging info
                # n_frames_anon += 1
                # sum_gpu_ms += (gpu_ms or 0.0)
                # sum_pix_ms += pix_ms

                # if n_frames_anon % 120 == 0:  # ~4s @30FPS
                #     avg_gpu = sum_gpu_ms / max(1, n_frames_anon)
                #     avg_pix = sum_pix_ms / max(1, n_frames_anon)
                #     log.info(
                #         "[anon] avg gpu_ms=%.2f avg pix_ms=%.2f (n=%d)",
                #         avg_gpu,
                #         avg_pix,
                #         n_frames_anon,
                #     )

            # pass along processed frame
            try:
                out_q.put_nowait((ts, frame))
            except Full:
                try:
                    out_q.get_nowait()
                except Empty:
                    pass
                try:
                    out_q.put_nowait((ts, frame))
                except Full:
                    pass

    except KeyboardInterrupt:
        log.info("[anon] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        log.exception(f"[anon] unexpected error: {e}")
    finally:
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        if cuda_ctx is not None:
            try:
                cuda_ctx.pop()
                log.info("[anon] CUDA context released.")
            except Exception:
                pass
        log.info("[anon] exit.")


def _build_anonymizer(cfg: dict):
    """
    Build an anonymizer callable that runs SCRFD TensorRT inference
    and applies pixelation to detected faces.

    Parameters
    ----------
    cfg : dict
        Keys include engine_path, input_name, size, conf, topk, max_dets,
        pixel_blocks, pixel_margin, pixel_max_faces, pixel_noise.

    Returns
    -------
    Callable[[any], Tuple[any, Optional[list], Optional[float]]]
        A callable(frame_bgr) -> (frame_bgr, dets, gpu_ms)
    """

    class _Anon:
        def __init__(self, cfg):
            self.inf = TRTInfer(
                engine_path=cfg["engine_path"],
                input_name=cfg.get("input_name", "input.1"),
                size=cfg.get("size", 640),
            )
            self.blocks = int(cfg.get("pixel_blocks", 8))
            self.margin = float(cfg.get("pixel_margin", 0.25))
            self.max_faces = int(cfg.get("pixel_max_faces", 32))
            self.noise = float(cfg.get("pixel_noise", 0.0))
            self.conf = float(cfg.get("conf", CONF_THRES))
            self.topk = int(cfg.get("topk", TOPK_PER_LEVEL))
            self.max_dets = int(cfg.get("max_dets", MAX_DETS))

        def __call__(self, frame_bgr):
            self.inf.conf_thres = self.conf
            self.inf.topk_per_level = self.topk
            self.inf.max_dets = self.max_dets
            dets, gpu_ms = self.inf.infer(frame_bgr)
            if dets is not None and len(dets) > 0:
                apply_pixelation(
                    frame_bgr,
                    dets,
                    margin=self.margin,
                    blocks=self.blocks,
                    max_faces=self.max_faces,
                    noise_sigma=self.noise,
                )
            return frame_bgr, dets, gpu_ms

    return _Anon(cfg)


class VideoStreamBlurFace:
    """
    Video pipeline that captures frames in one process, optionally anonymizes
    faces in another process (via SCRFD TensorRT + pixelation), then encodes
    frames as base64 JPEG and dispatches them to registered callbacks.

    Parameters
    ----------
    frame_callbacks : Optional[Iterable[Callable[[str], None]]], optional
        List (or single callable) receiving base64-encoded JPEG strings.
    fps : int, optional
        Target output frames per second, by default 30.
    resolution : Tuple[int, int], optional
        Desired frame resolution (width, height), by default (640, 480).
    jpeg_quality : int, optional
        JPEG quality (0–100), by default 70.
    blur_enabled : bool, optional
        If True and an engine path is supplied, apply face anonymization.
    blur_conf : float, optional
        Detection confidence threshold, by default 0.5.
    scrfd_engine : Optional[str], optional
        Path to a TensorRT engine file. If None, anonymization is disabled.
    scrfd_size : int, optional
        Model input size (square), by default 640.
    scrfd_input : str, optional
        Engine input name, by default "input.1".
    pixel_blocks : int, optional
        Pixelation block size parameter, by default 8.
    pixel_margin : float, optional
        Margin multiplier around detected boxes, by default 0.25.
    pixel_max_faces : int, optional
        Max number of faces to pixelate, by default 32.
    pixel_noise : float, optional
        Gaussian noise sigma added to pixelated regions, by default 0.0.
    draw_boxes : bool, optional
        Draw detection boxes (debug), by default False.
    queue_size_raw : int, optional
        Max size for the raw frames queue (capture→anon), by default 4.
    queue_size_proc : int, optional
        Max size for the processed frames queue (anon→main), by default 4.
    buffer_frames : int, optional
        Desired capture driver buffer size, by default 1.
    camera : Optional[str], optional
        Explicit camera index/path; if None, auto-pick as in the reference style.
    log_queue : Optional[mp.Queue], optional
        Logging queue from runtime.logging.setup_logging_mp_main(). If provided,
        child processes forward logs to the main process.
    """

    def __init__(
        self,
        frame_callbacks: Optional[Iterable[Callable[[str], None]]] = None,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
        jpeg_quality: int = 70,
        blur_enabled: bool = True,
        blur_conf: float = 0.5,
        scrfd_engine: Optional[str] = None,
        scrfd_size: int = 640,
        scrfd_input: str = "input.1",
        pixel_blocks: int = 8,
        pixel_margin: float = 0.25,
        pixel_max_faces: int = 32,
        pixel_noise: float = 0.0,
        draw_boxes: bool = False,
        queue_size_raw: int = 4,
        queue_size_proc: int = 4,
        buffer_frames: int = 1,
        camera: Optional[str] = None,
        log_queue: Optional[mp.Queue] = None,
    ):
        self.fps = int(fps)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.encode_quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

        # Callbacks
        self._cb_lock = threading.Lock()
        if frame_callbacks is None:
            self.frame_callbacks: List[Callable[[str], None]] = []
        elif callable(frame_callbacks):
            self.frame_callbacks = [frame_callbacks]  # type: ignore[list-item]
        else:
            self.frame_callbacks = list(frame_callbacks)

        # Async loop for scheduling async callbacks (if any)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()
        logger.debug("Starting background event loop for video streaming.")

        # Camera + Queues
        self.cam = camera if camera is not None else _pick_camera()
        self.q_raw = mp.Queue(maxsize=int(queue_size_raw))
        self.q_proc = mp.Queue(maxsize=int(queue_size_proc))

        # Anonymizer config
        self.scrfd_cfg = dict(
            engine_path=scrfd_engine,
            size=int(scrfd_size),
            input_name=scrfd_input,
            conf=float(blur_conf) if blur_conf is not None else CONF_THRES,
            topk=int(TOPK_PER_LEVEL),
            max_dets=int(MAX_DETS),
            pixel_blocks=int(pixel_blocks),
            pixel_margin=float(pixel_margin),
            pixel_max_faces=int(pixel_max_faces),
            pixel_noise=float(pixel_noise),
        )
        self.blur_enabled = bool(blur_enabled)
        self.draw_boxes = bool(draw_boxes)
        self.buffer_frames = int(buffer_frames)

        # Processes and control
        self.p_cap: Optional[mp.Process] = None
        self.p_anon: Optional[mp.Process] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._running = mp.Value("b", False)

        # Logging queue for children
        self.log_queue = log_queue

    # ----------------------------
    # Public API
    # ----------------------------
    def register_frame_callback(self, cb: Callable[[str], None]) -> None:
        """
        Register a per-frame callback.

        Parameters
        ----------
        cb : Callable[[str], None]
            Callback receiving a base64-encoded JPEG string.
        """
        if cb is None:
            logger.warning("Frame callback is None, not registering")
            return
        with self._cb_lock:
            if cb not in self.frame_callbacks:
                self.frame_callbacks.append(cb)
                logger.info("Registered new frame callback")
            else:
                logger.warning("Frame callback already registered")

    def unregister_frame_callback(self, cb: Callable[[str], None]) -> None:
        """
        Unregister a previously added callback.

        Parameters
        ----------
        cb : Callable[[str], None]
            Callback to remove. No-op if not present.
        """
        with self._cb_lock:
            try:
                self.frame_callbacks.remove(cb)
                logger.info("Unregistered frame callback")
            except ValueError:
                logger.warning("Attempted to unregister a non-registered callback")

    def start(self) -> None:
        """
        Start capture/anonymize worker processes and the drain thread.
        """
        if self._running.value:
            logger.warning("[main] start() called but already running.")
            return

        self._running.value = True
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

        self.p_cap = mp.Process(
            target=proc_capture,
            args=(
                self.q_raw,
                self.cam,
                self.resolution,
                self.fps,
                self.buffer_frames,
                self.log_queue,  # pass logging queue to child
            ),
            daemon=True,
            name="CaptureProc",
        )
        self.p_anon = mp.Process(
            target=proc_anonymize,
            args=(
                self.q_raw,
                self.q_proc,
                self.scrfd_cfg,
                self.blur_enabled,
                self.draw_boxes,
                self.log_queue,  # pass logging queue to child
            ),
            daemon=True,
            name="AnonProc",
        )

        self.p_cap.start()
        logger.info("[main] Capture process started (pid=%s).", self.p_cap.pid)
        self.p_anon.start()
        logger.info("[main] Anonymize process started (pid=%s).", self.p_anon.pid)

        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="DrainThread"
        )
        self._drain_thread.start()
        logger.info("[main] Drain thread started. VideoStream running.")

    def stop(self, join_timeout: float = 1.0) -> None:
        """
        Stop workers and drain thread, and release resources.

        Parameters
        ----------
        join_timeout : float, optional
            Time (seconds) to wait for processes/threads to join before terminate, by default 1.0
        """
        if not self._running.value:
            logger.warning("[main] stop() called but not running.")
            return

        self._running.value = False

        # Signal workers
        for q in (self.q_raw, self.q_proc):
            try:
                q.put_nowait(SENTINEL)
            except Full:
                pass

        # Join processes
        if self.p_cap and self.p_cap.is_alive():
            self.p_cap.join(timeout=join_timeout)
            if self.p_cap.is_alive():
                logger.warning("[main] CaptureProc did not exit in time; terminating.")
                self.p_cap.terminate()

        if self.p_anon and self.p_anon.is_alive():
            self.p_anon.join(timeout=join_timeout)
            if self.p_anon.is_alive():
                logger.warning("[main] AnonProc did not exit in time; terminating.")
                self.p_anon.terminate()

        # Join drain thread
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=join_timeout)

        # Stop background loop
        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass

        logger.info("[main] VideoStream stopped.")

    # ----------------------------
    # Internals
    # ----------------------------
    def _start_loop(self) -> None:
        """Start the asyncio event loop in a dedicated thread."""
        asyncio.set_event_loop(self.loop)
        logger.debug("Starting background asyncio event loop.")
        self.loop.run_forever()

    def _dispatch(self, b64_jpeg: str) -> None:
        """
        Dispatch a base64-encoded JPEG frame to all registered callbacks.

        Parameters
        ----------
        b64_jpeg : str
            Base64-encoded JPEG bytes.
        """
        with self._cb_lock:
            callbacks = tuple(self.frame_callbacks)

        for cb in callbacks:
            try:
                result = cb(b64_jpeg)
                if inspect.isawaitable(result):
                    asyncio.run_coroutine_threadsafe(result, self.loop)
            except Exception as e:
                logger.error("Frame callback raised: %s", e, exc_info=True)

    def _drain_loop(self) -> None:
        """
        Main-process drain loop.
        - Pull processed frames
        - JPEG encode + base64
        - Dispatch to callbacks
        - Pace to ~fps
        """
        frame_time = 1.0 / max(1, self.fps)
        last = time.perf_counter()

        logger.info("[main] drain loop starting at ~%d FPS.", self.fps)
        while self._running.value:
            try:
                ts, frame = self.q_proc.get(timeout=0.5)
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                logger.info("[main] drain loop received sentinel; exiting.")
                break

            ok, buf = cv2.imencode(".jpg", frame, self.encode_quality)
            if ok:
                self._dispatch(base64.b64encode(buf).decode("utf-8"))
            else:
                logger.error("[main] JPEG encode failed; dropping frame.")

            elapsed = time.perf_counter() - last
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last = time.perf_counter()
