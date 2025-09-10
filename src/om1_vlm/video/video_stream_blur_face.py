# -*- coding: utf-8 -*-
# om1_vlm/video/video_stream_blur_face.py
"""
Video capture + optional face anonymization (pixelation) pipeline.

- Capture process (V4L2 on Linux) reads frames and pushes to a queue
- Anonymization process (optional, TensorRT SCRFD) pixelates faces
- Main-thread drain loop encodes frames to JPEG (base64) and dispatches callbacks
  with a latest-frame-wins policy to keep latency low.
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
import numpy as np  # NEW: for buffer conversions

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

try:
    import pycuda.driver as cuda
except ImportError:
    print("pycuda not found, face anonymization will be disabled.")
    cuda = None

# Keep logger naming consistent with your other modules
root_package_name = __name__.split(".")[0] if "." in __name__ else __name__
logger = logging.getLogger(root_package_name)

SENTINEL = ("__STOP__", None)


def proc_capture(
    out_q: mp.Queue,
    cam: str,
    res: Tuple[int, int],
    fps: int,
    buffer_frames: int,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Capture frames from a camera in a dedicated process and push them to a queue.
    """
    setup_logging("odom_processor", logging_config=logging_config)
    cap = None

    try:
        # Prefer V4L2 on Linux; macOS ignores this backend id.
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            logging.warning(f"[cap] failed to open {cam}, trying common fallbacks.")

        if not cap or not cap.isOpened():
            logging.error(f"[cap] cannot open camera {cam}")
            return

        # Apply settings (best-effort)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, fps)

        try:
            # MJPG can reduce CPU usage (toggle off to compare latency)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_frames)))
        except Exception:
            pass

        # Warn if camera doesn't honor the requested resolution/fps
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if (actual_w, actual_h) != (res[0], res[1]):
            logging.warning(
                f"[cap] Camera doesn't support resolution {res}. Using {(actual_w, actual_h)} instead."
            )
        logging.info(f"[cap] camera reports FPS={actual_fps:.2f}")

        frame_time = 1.0 / max(1, fps)
        last = time.perf_counter()

        logging.info(f"[cap] capture loop starting at ~{fps} FPS, device={cam}")
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.error("[cap] error reading frame; retrying shortly.")
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
        logging.info("[cap] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        logging.exception(f"[cap] unexpected error: {e}")
    finally:
        try:
            if cap is not None:
                cap.release()
                logging.info("[cap] released video capture device")
        except Exception:
            pass
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        logging.info("[cap] exit.")


def proc_anonymize(
    in_q: mp.Queue,
    out_q: mp.Queue,
    scrfd_cfg: dict,
    blur_enabled: bool,
    draw_boxes: bool,
    logging_config: Optional[LoggingConfig] = None,
) -> None:
    """
    Optional anonymizer process: applies face detection & pixelation, then forwards frames.
    """
    setup_logging("odom_processor", logging_config=logging_config)

    anonymizer = None
    cuda_ctx = None
    sum_gpu_ms = 0.0
    sum_pix_ms = 0.0
    n_frames_anon = 0

    if cuda is None:
        blur_enabled = False
        logging.error("[anon] pycuda not found, disabling anonymization.")

    try:
        if blur_enabled and scrfd_cfg.get("engine_path"):
            cuda.init()
            dev_id = int(os.environ.get("OM1_CUDA_DEVICE", "0"))
            cuda_ctx = cuda.Device(dev_id).make_context()
            logging.info(f"[anon] CUDA context created on device {dev_id}")

            try:
                anonymizer = _build_anonymizer(scrfd_cfg)
                logging.info("[anon] anonymizer initialized.")
            except Exception:
                try:
                    cuda_ctx.pop()
                except Exception:
                    pass
                logging.exception("[anon] failed to initialize anonymizer.")
                raise

        logging.info("[anon] anonymize loop starting (enabled=%s).", bool(anonymizer))
        while True:
            try:
                ts, frame = in_q.get(timeout=1.0)
            except Empty:
                continue

            if (ts, frame) == SENTINEL:
                logging.info("[anon] received sentinel; exiting loop.")
                break

            if blur_enabled and anonymizer is not None:
                t0 = time.perf_counter()
                frame, dets, gpu_ms = anonymizer(frame)
                t1 = time.perf_counter()
                pix_ms = (t1 - t0) * 1000.0 - (gpu_ms or 0.0)
                if pix_ms < 0:
                    pix_ms = 0.0

                if draw_boxes and dets is not None:
                    draw_dets(frame, dets)

                n_frames_anon += 1
                sum_gpu_ms += gpu_ms or 0.0
                sum_pix_ms += pix_ms

                if n_frames_anon % 120 == 0:
                    avg_gpu = sum_gpu_ms / max(1, n_frames_anon)
                    avg_pix = sum_pix_ms / max(1, n_frames_anon)
                    logging.info(
                        "[anon] avg gpu_ms=%.2f avg pix_ms=%.2f (n=%d)",
                        avg_gpu,
                        avg_pix,
                        n_frames_anon,
                    )

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
        logging.info("[anon] received KeyboardInterrupt; shutting down.")
    except Exception as e:
        logging.exception(f"[anon] unexpected error: {e}")
    finally:
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        if cuda_ctx is not None:
            try:
                cuda_ctx.pop()
                logging.info("[anon] CUDA context released.")
            except Exception:
                pass
        logging.info("[anon] exit.")


def _build_anonymizer(cfg: dict):
    """
    Build an anonymizer callable that runs SCRFD TensorRT inference
    and applies pixelation to detected faces.
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
    frames as base64 JPEG and dispatches to registered callbacks.
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
        queue_size_raw: int = 1,
        queue_size_proc: int = 1,
        buffer_frames: int = 1,
        use_turbojpeg: bool = True,         # NEW: allow disabling at runtime
        raw_frame_callbacks: Optional[Iterable[Callable[[np.ndarray], None]]] = None,  # NEW
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

        # NEW: raw callbacks (receive numpy frames, no JPEG/base64)
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

        # Processes and control
        self.p_cap: Optional[mp.Process] = None
        self.p_anon: Optional[mp.Process] = None
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

    # NEW: raw callback API
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
        """Start capture/anonymize worker processes and the drain thread."""
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
                get_logging_config(),
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
                get_logging_config(),
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
        """Stop workers and drain thread, and release resources."""
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
        """Dispatch a base64-encoded JPEG frame to all registered callbacks."""
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
        """Dispatch a raw numpy frame to raw callbacks (no encode/base64)."""
        with self._cb_lock:
            raw_cbs = tuple(self.raw_callbacks)
        for cb in raw_cbs:
            try:
                cb(frame_bgr)
            except Exception as e:
                logger.error("Raw frame callback raised: %s", e, exc_info=True)

    # --- JPEG encoding helper (TurboJPEG preferred) ---
    def _encode_jpeg(self, frame_bgr) -> Optional[bytes]:
        """
        Encode BGR frame to JPEG bytes using TurboJPEG if available,
        else cv2.imencode fallback. Returns None on failure.
        """
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
        Main-process drain loop with latest-frame-wins:
        - Pull processed frames, drain queue to newest
        - Dispatch raw frame (no encoding) for local preview (optional)
        - JPEG encode + base64 for standard callbacks
        - Pace only when ahead (avoid extra delay when behind)
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

            # Dispatch raw frame first (fast local preview, no encode)
            self._dispatch_raw(frame)

            # Encode & dispatch base64 JPEG for normal callbacks / WS
            jpeg_bytes = self._encode_jpeg(frame)
            if jpeg_bytes is not None:
                self._dispatch(base64.b64encode(jpeg_bytes).decode("utf-8"))
            else:
                logger.error("[main] JPEG encode failed; dropping frame.")
                continue

            # Pace only if we're ahead (frame is fresh & loop was faster than frame_time)
            now = time.perf_counter()
            elapsed = now - last
            # If frame age is small and loop was faster than target, sleep a bit
            if (now - ts) < frame_time and elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last = time.perf_counter()
