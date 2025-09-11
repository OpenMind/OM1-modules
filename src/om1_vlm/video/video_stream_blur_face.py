# -*- coding: utf-8 -*-
# om1_vlm/video/video_stream_blur_face.py
"""
Video capture + optional face anonymization (pixelation) pipeline.

Architecture (low-latency):
- One worker subprocess does BOTH camera capture and (optional) anonymization.
- Main process drain thread encodes to JPEG (TurboJPEG preferred) and dispatches:
  * raw callbacks: numpy BGR frames (no encode/base64) for local preview
  * frame callbacks: base64 JPEG strings (e.g., for WebSocket streaming)

Policies:
- Capture side: NO throttling; let device drive the pace.
- Queues: tiny size (default 1). When full, drop oldest (latest-frame-wins).
- Drain side: only sleep when we're AHEAD of target FPS; if behind, never sleep.
"""

import asyncio
import base64
import inspect
import logging
import multiprocessing as mp
import os
import platform
import threading
import time
from queue import Empty, Full
from typing import Callable, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from om1_utils.logging import LoggingConfig, get_logging_config, setup_logging

from ..anonymizationSys.scrfd_trt_pixelate import (
    CONF_THRES,
    MAX_DETS,
    TOPK_PER_LEVEL,
    TRTInfer,
    apply_pixelation,
    draw_dets,
)
from .video_utils import enumerate_video_devices

# --- TurboJPEG (fast JPEG) ---
try:
    from turbojpeg import TurboJPEG, TJPF, TJSAMP
    _HAVE_TURBOJPEG = True
except Exception:
    _HAVE_TURBOJPEG = False

# --- CUDA for SCRFD TensorRT ---
try:
    import pycuda.driver as cuda
except ImportError:
    print("pycuda not found, face anonymization will be disabled.")
    cuda = None

# Keep logger naming consistent with your other modules
root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

SENTINEL = ("__STOP__", None)


# ---------------------------------------------------------------------
# Combined worker process: Camera + (optional) anonymization
# ---------------------------------------------------------------------
def proc_cam_ml(
    out_q: mp.Queue,
    cam: str,
    res: Tuple[int, int],
    target_fps: int,
    buffer_frames: int,
    scrfd_cfg: dict,
    blur_enabled: bool,
    draw_boxes: bool,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """Single worker process that captures frames and optionally anonymizes them,
    then publishes the (timestamp, frame_bgr) to out_q with latest-frame-wins.
    """
    setup_logging("odom_processor", logging_config=logging_config)
    try:
        cv2.setNumThreads(1)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass

    cap = None
    anonymizer = None
    cuda_ctx = None

    try:
        # Open camera (prefer V4L2 on Linux; ignored on macOS)
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap or not cap.isOpened():
            logging.error(f"[camml] cannot open camera {cam}")
            return

        # Best-effort camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, target_fps)

        # If latency seems high, try toggling MJPG off/on to compare:
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_frames)))
        except Exception:
            pass

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"[camml] camera opened {cam} -> {(actual_w, actual_h)} fps={actual_fps:.2f}")

        # Init anonymizer (same process; keeps one CUDA context)
        if blur_enabled and scrfd_cfg.get("engine_path") and cuda is not None:
            try:
                cuda.init()
                dev_id = int(os.environ.get("OM1_CUDA_DEVICE", "0"))
                cuda_ctx = cuda.Device(dev_id).make_context()
                anonymizer = _build_anonymizer(scrfd_cfg)
                logging.info(f"[camml] anonymizer ready on CUDA device {dev_id}")
            except Exception:
                logging.exception("[camml] anonymizer init failed; running without blur")
                anonymizer = None
        elif blur_enabled and cuda is None:
            logging.error("[camml] pycuda not found; blur disabled")

        # Telemetry
        n = 0
        sum_gpu_ms = 0.0
        sum_pix_ms = 0.0

        logging.info("[camml] loop start")
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.error("[camml] cap.read() failed; retry shortly")
                time.sleep(0.02)
                continue

            ts = time.time()

            if blur_enabled and anonymizer is not None:
                t0 = time.perf_counter()
                frame, dets, gpu_ms = anonymizer(frame)
                t1 = time.perf_counter()
                pix_ms = (t1 - t0) * 1000.0 - (gpu_ms or 0.0)
                if pix_ms < 0:
                    pix_ms = 0.0

                n += 1
                sum_gpu_ms += (gpu_ms or 0.0)
                sum_pix_ms += pix_ms
                if n % 120 == 0:
                    logging.info(
                        "[camml] avg gpu_ms=%.2f avg pix_ms=%.2f (n=%d)",
                        sum_gpu_ms / max(1, n),
                        sum_pix_ms / max(1, n),
                        n,
                    )
                if draw_boxes and dets is not None:
                    draw_dets(frame, dets)

            # Latest-frame-wins: if full, drop one stale then push
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

            # IMPORTANT: no throttling here; let drain control pacing

    except KeyboardInterrupt:
        logging.info("[camml] KeyboardInterrupt; exit")
    except Exception as e:
        logging.exception(f"[camml] unexpected error: {e}")
    finally:
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        if cuda_ctx is not None:
            try:
                cuda_ctx.pop()
            except Exception:
                pass
        logging.info("[camml] exit.")


def _build_anonymizer(cfg: dict):
    """Build an anonymizer callable that runs SCRFD TensorRT and applies pixelation."""
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


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
class VideoStreamBlurFace:
    """
    Video pipeline that:
      - spawns ONE worker process (camera + optional anonymization),
      - drains frames in the main process (encode + dispatch),
      - keeps latency low with latest-frame-wins throughout.
    """

    def __init__(
        self,
        frame_callbacks: Optional[Iterable[Callable[[str], None]]] = None,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480),
        jpeg_quality: int = 70,
        device_index: int = 0,
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
        queue_size_proc: int = 1,
        buffer_frames: int = 1,
        use_turbojpeg: bool = True,
        raw_frame_callbacks: Optional[Iterable[Callable[[np.ndarray], None]]] = None,
    ):
        self.fps = int(fps)
        try:
            cv2.setNumThreads(1)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass

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

        if raw_frame_callbacks is None:
            self.raw_callbacks: List[Callable[[np.ndarray], None]] = []
        elif callable(raw_frame_callbacks):
            self.raw_callbacks = [raw_frame_callbacks]  # type: ignore[list-item]
        else:
            self.raw_callbacks = list(raw_frame_callbacks)

        # Async loop for scheduling async callbacks (if any)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()
        logger.debug("Starting background event loop for video streaming.")

        # Camera device selection
        devices = enumerate_video_devices()
        if platform.system() == "Darwin":
            camindex = 0 if devices else 0
        else:
            camindex = f"/dev/video{devices[0][0]}" if devices else "/dev/video0"
        if device_index != 0:
            if platform.system() == "Darwin":
                camindex = device_index
            else:
                camindex = f"/dev/video{device_index}"
        self.cam = camindex

        # Single queue from worker → main drain
        self.q_proc = mp.Queue(maxsize=int(queue_size_proc))

        # Anonymizer config for worker
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

        # JPEG encoder (TurboJPEG preferred)
        self._jpeg_quality = int(jpeg_quality)
        self._use_turbojpeg = _HAVE_TURBOJPEG and bool(use_turbojpeg)
        self._jpeg = None
        if self._use_turbojpeg:
            try:
                self._jpeg = TurboJPEG()
            except Exception:
                logger.warning("TurboJPEG load failed, falling back to cv2.imencode")
                self._use_turbojpeg = False

        # Process and control
        self.p_camml: Optional[mp.Process] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._running = mp.Value("b", False)

    # ----------------------------
    # Public API
    # ----------------------------
    def register_frame_callback(self, cb: Callable[[str], None]) -> None:
        """Register a per-frame (base64 JPEG) callback."""
        if cb is None:
            logger.warning("Frame callback is None, not registering")
            return
        with self._cb_lock:
            if cb not in self.frame_callbacks:
                self.frame_callbacks.append(cb)
                logger.info("Registered new frame callback")
            else:
                logger.warning("Frame callback already registered")

    def register_raw_frame_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register a per-frame raw (numpy BGR) callback."""
        if cb is None:
            logger.warning("Raw frame callback is None, not registering")
            return
        with self._cb_lock:
            if cb not in self.raw_callbacks:
                self.raw_callbacks.append(cb)
                logger.info("Registered new raw frame callback")
            else:
                logger.warning("Raw frame callback already registered")

    def unregister_frame_callback(self, cb: Callable[[str], None]) -> None:
        """Unregister a previously added base64 JPEG callback."""
        with self._cb_lock:
            try:
                self.frame_callbacks.remove(cb)
                logger.info("Unregistered frame callback")
            except ValueError:
                logger.warning("Attempted to unregister a non-registered callback")

    def unregister_raw_frame_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Unregister a previously added raw frame callback."""
        with self._cb_lock:
            try:
                self.raw_callbacks.remove(cb)
                logger.info("Unregistered raw frame callback")
            except ValueError:
                logger.warning("Attempted to unregister a non-registered callback")

    def start(self) -> None:
        """Start worker process (cam+ML) and the drain thread."""
        if self._running.value:
            logger.warning("[main] start() called but already running.")
            return

        self._running.value = True
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

        self.p_camml = mp.Process(
            target=proc_cam_ml,
            args=(
                self.q_proc,          # worker publishes directly to q_proc
                self.cam,
                self.resolution,
                self.fps,
                self.buffer_frames,
                self.scrfd_cfg,
                self.blur_enabled,
                self.draw_boxes,
                get_logging_config(),
            ),
            daemon=True,
            name="CamMLProc",
        )
        self.p_camml.start()
        logger.info("[main] CamML process started (pid=%s).", self.p_camml.pid)

        self._drain_thread = threading.Thread(
            target=self._drain_loop, daemon=True, name="DrainThread"
        )
        self._drain_thread.start()
        logger.info("[main] Drain thread started. VideoStream running.")

    def stop(self, join_timeout: float = 1.0) -> None:
        """Stop worker and drain thread, and release resources."""
        if not self._running.value:
            logger.warning("[main] stop() called but not running.")
            return

        self._running.value = False

        # Signal worker
        try:
            self.q_proc.put_nowait(SENTINEL)
        except Full:
            pass

        # Join worker
        if self.p_camml and self.p_camml.is_alive():
            self.p_camml.join(timeout=join_timeout)
            if self.p_camml.is_alive():
                logger.warning("[main] CamMLProc did not exit in time; terminating.")
                self.p_camml.terminate()

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
        """Dispatch base64 JPEG to all registered callbacks."""
        with self._cb_lock:
            callbacks = tuple(self.frame_callbacks)
        for cb in callbacks:
            try:
                result = cb(b64_jpeg)
                if inspect.isawaitable(result):
                    asyncio.run_coroutine_threadsafe(result, self.loop)
            except Exception as e:
                logger.error("Frame callback raised: %s", e, exc_info=True)

    def _dispatch_raw(self, frame_bgr: np.ndarray) -> None:
        """Dispatch raw numpy frame (no encode/base64) to raw callbacks."""
        with self._cb_lock:
            raw_cbs = tuple(self.raw_callbacks)
        for cb in raw_cbs:
            try:
                cb(frame_bgr)
            except Exception as e:
                logger.error("Raw frame callback raised: %s", e, exc_info=True)

    def _encode_jpeg(self, frame_bgr) -> Optional[bytes]:
        """Encode BGR frame to JPEG bytes (TurboJPEG if available, else cv2)."""
        try:
            if self._use_turbojpeg and self._jpeg is not None:
                return self._jpeg.encode(
                    frame_bgr,
                    quality=self._jpeg_quality,
                    pixel_format=TJPF.BGR,
                    jpeg_subsample=TJSAMP.SAMP_420,
                )
            else:
                ok, buf = cv2.imencode(
                    ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
                )
                if not ok:
                    return None
                return buf.tobytes()
        except Exception:
            logger.exception("[main] JPEG encode failed")
            return None

    def _drain_loop(self) -> None:
        """
        Drain loop (main process):
        - Pull frames, drain queue to newest (latest-frame-wins)
        - Dispatch raw frame first (for local preview)
        - Encode to JPEG and dispatch base64
        - Only sleep when we're ahead of target FPS
        """
        frame_time = 1.0 / max(1, self.fps)
        last = time.perf_counter()

        logger.info("[main] drain loop starting at ~%d FPS.", self.fps)
        while self._running.value:
            # Pull at least one frame
            try:
                ts, frame = self.q_proc.get(timeout=0.05)
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                logger.info("[main] drain loop received sentinel; exiting.")
                break

            # Drain queue: keep the most recent frame, drop stale
            while True:
                try:
                    ts2, frame2 = self.q_proc.get_nowait()
                    if (ts2, frame2) == SENTINEL:
                        logger.info("[main] drain got sentinel during drain; exiting.")
                        ts, frame = ts2, frame2
                        break
                    ts, frame = ts2, frame2
                except Empty:
                    break
            if (ts, frame) == SENTINEL:
                break

            # Raw dispatch first (no encode)
            self._dispatch_raw(frame)

            # JPEG encode + base64 dispatch
            jpeg_bytes = self._encode_jpeg(frame)
            if jpeg_bytes is not None:
                self._dispatch(base64.b64encode(jpeg_bytes).decode("utf-8"))
            else:
                logger.error("[main] JPEG encode failed; dropping frame.")
                continue

            # Pace only if we're ahead (frame is fresh & loop faster than target)
            # now = time.perf_counter()
            # elapsed = now - last
            # if (now - ts) < frame_time and elapsed < frame_time:
            #     time.sleep(frame_time - elapsed)
            # last = time.perf_counter()
