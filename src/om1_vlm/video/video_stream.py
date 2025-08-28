# -*- coding: utf-8 -*-
# mp_videostream.py
"""
cd OM1-modules
export PYTHONPATH="$PWD/src:$PYTHONPATH"      # Key: ensure src is on PYTHONPATH

Multiprocess video pipeline for OM1:

    [Process A: Capture]  camera -> raw frames  ─┐
                                                 ├─> Queue (raw)
    [Process B: Anonym.] raw -> SCRFD+pixelate  ─┘
             │
             └─> Queue (processed) -> [Main Thread: JPEG encode + callbacks]


Design goals:
- Isolate blocking I/O (camera) and GPU/TensorRT work in their OWN processes.
- Keep the main app responsive (e.g., VLM and other subsystems).

"""



import os, cv2, time, base64, inspect, asyncio, logging, platform, multiprocessing as mp
from queue import Full, Empty
import threading
from typing import Optional, Callable, List, Tuple

# your imports
from .video_utils import enumerate_video_devices
from om1_ml.anonymizationSys.scrfd_trt_pixelate import (
    TRTInfer, apply_pixelation, draw_dets, CONF_THRES, TOPK_PER_LEVEL, MAX_DETS,
)

logger = logging.getLogger(__name__)
SENTINEL = ("__STOP__", None)

def _pick_camera() -> str:
    devs = enumerate_video_devices()
    if platform.system() == "Darwin":
        return 0 if devs else 0
    return f"/dev/video{devs[0][0]}" if devs else "/dev/video0"

# ----------------------------
# Process A: Capture
# ----------------------------
def proc_capture(out_q: mp.Queue, cam: str, res: Tuple[int,int], fps: int, buffer_frames: int):
    try:
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            logger.error(f"[cap] cannot open {cam}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS,          fps)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, buffer_frames))
        except Exception: pass
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception: pass

        frame_time = 1.0 / max(1, fps)
        last = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05); continue
            ts = time.time()
            pkt = (ts, frame)

            # non-blocking put with drop policy
            try:
                out_q.put_nowait(pkt)
            except Full:
                # optional: drop one stale frame to make room, then put
                try: out_q.get_nowait()
                except Empty: pass
                try: out_q.put_nowait(pkt)
                except Full: pass

            # pace
            elapsed = time.perf_counter() - last
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last = time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        try: cap.release()
        except Exception: pass
        # signal downstream to stop
        try: out_q.put_nowait(SENTINEL)
        except Full: pass
        logger.info("[cap] exit.")

# ----------------------------
# Process B: Anonymize (SCRFD+pixelation)
# ----------------------------
def proc_anonymize(in_q: mp.Queue, out_q: mp.Queue,
                   scrfd_cfg: dict, blur_enabled: bool,
                   draw_boxes: bool):
    # IMPORTANT: build TRTInfer in THIS process
    anonymizer = None
    if blur_enabled and scrfd_cfg.get("engine_path"):
        anonymizer = _build_anonymizer(scrfd_cfg)

    try:
        while True:
            try:
                ts, frame = in_q.get(timeout=1.0)
            except Empty:
                continue
            if (ts, frame) == SENTINEL:
                break

            if blur_enabled and anonymizer is not None:
                frame, dets, _gpu_ms = anonymizer(frame)
                if draw_boxes:
                    draw_dets(frame, dets)

            # pass along processed frame
            try:
                out_q.put_nowait((ts, frame))
            except Full:
                # drop policy
                try: out_q.get_nowait()
                except Empty: pass
                try: out_q.put_nowait((ts, frame))
                except Full: pass
    finally:
        # cascade stop
        try: out_q.put_nowait(SENTINEL)
        except Full: pass
        logger.info("[anon] exit.")

def _build_anonymizer(cfg: dict):
    class _Anon:
        def __init__(self, cfg):
            self.inf = TRTInfer(engine_path=cfg["engine_path"],
                                input_name=cfg.get("input_name","input.1"),
                                size=cfg.get("size",640),
                                verbose=cfg.get("verbose",False))
            self.blocks = cfg.get("pixel_blocks", 8)
            self.margin = cfg.get("pixel_margin", 0.25)
            self.max_faces = cfg.get("pixel_max_faces", 32)
            self.noise = cfg.get("pixel_noise", 0.0)
            self.conf = float(cfg.get("conf", CONF_THRES))
            self.topk = int(cfg.get("topk", TOPK_PER_LEVEL))
            self.max_dets = int(cfg.get("max_dets", MAX_DETS))

        def __call__(self, frame_bgr):
            self.inf.conf_thres = self.conf
            self.inf.topk_per_level = self.topk
            self.inf.max_dets = self.max_dets
            dets, gpu_ms = self.inf.infer(frame_bgr)
            if dets is not None and len(dets) > 0:
                apply_pixelation(frame_bgr, dets,
                                 margin=self.margin, blocks=self.blocks,
                                 max_faces=self.max_faces, noise_sigma=self.noise)
            return frame_bgr, dets, gpu_ms
    return _Anon(cfg)

# ----------------------------
# Main-side wrapper
# ----------------------------
class VideoStream:
    def __init__(self,
                 frame_callbacks: Optional[List[Callable[[str], None]]] = None,
                 fps: int = 30,
                 resolution: Tuple[int,int] = (640,480),
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
                 buffer_frames: int = 1):
        self.fps = fps
        self.resolution = resolution
        self.encode_quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

        self.frame_callbacks = frame_callbacks or []
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

        self.cam = _pick_camera()
        self.q_raw  = mp.Queue(maxsize=queue_size_raw)
        self.q_proc = mp.Queue(maxsize=queue_size_proc)

        self.scrfd_cfg = dict(
            engine_path=scrfd_engine,
            size=scrfd_size,
            input_name=scrfd_input,
            conf=float(blur_conf) if blur_conf is not None else CONF_THRES,
            topk=TOPK_PER_LEVEL,
            max_dets=MAX_DETS,
            pixel_blocks=pixel_blocks,
            pixel_margin=pixel_margin,
            pixel_max_faces=pixel_max_faces,
            pixel_noise=pixel_noise,
            verbose=False,
        )
        self.blur_enabled = bool(blur_enabled)
        self.draw_boxes = draw_boxes
        self.buffer_frames = buffer_frames

        self.p_cap: Optional[mp.Process] = None
        self.p_anon: Optional[mp.Process] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._running = mp.Value('b', False)

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        if self._running.value:
            return
        self._running.value = True
        # IMPORTANT for macOS/Windows:
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

        self.p_cap = mp.Process(
            target=proc_capture,
            args=(self.q_raw, self.cam, self.resolution, self.fps, self.buffer_frames),
            daemon=True
        )
        self.p_anon = mp.Process(
            target=proc_anonymize,
            args=(self.q_raw, self.q_proc, self.scrfd_cfg, self.blur_enabled, self.draw_boxes),
            daemon=True
        )
        self.p_cap.start()
        self.p_anon.start()

        # main-side consumer: encode & callback
        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()
        logger.info("[main] VideoStream started.")

    def _dispatch(self, b64_jpeg: str):
        for cb in self.frame_callbacks:
            if inspect.iscoroutinefunction(cb):
                asyncio.run_coroutine_threadsafe(cb(b64_jpeg), self.loop)
            else:
                cb(b64_jpeg)

    def _drain_loop(self):
        """
        Main-process drain loop.

        Responsibilities
        ----------------
        - Pull (timestamp, processed_frame) from q_proc.
        - Encode each frame to JPEG (OpenCV).
        - Base64-encode and dispatch via registered callbacks.
        - Pace to ~fps to avoid flooding downstream.

        Exit
        ----
        - Breaks when SENTINEL is received or `_running` is cleared.
        """
        frame_time = 1.0 / max(1, self.fps)
        last = time.perf_counter()
        try:
            while self._running.value:
                try:
                    ts, frame = self.q_proc.get(timeout=0.5)
                except Empty:
                    continue
                if (ts, frame) == SENTINEL:
                    break

                ok, buf = cv2.imencode(".jpg", frame, self.encode_quality)
                if ok:
                    self._dispatch(base64.b64encode(buf).decode("utf-8"))

                elapsed = time.perf_counter() - last
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last = time.perf_counter()
        finally:
            logger.info("[main] drain exit.")

    def stop(self, join_timeout: float = 1.0):
        """
        Stop the entire pipeline and release resources.

        Steps
        -----
        1) Clear `_running` flag.
        2) Push SENTINEL to both queues to unblock any pending gets.
        3) Join child processes with a small timeout; terminate if still alive.
        4) Join the drain thread.
        5) Stop the asyncio loop thread.

        Parameters
        ----------
        join_timeout : float
            Seconds to wait for processes/threads to join before forcing termination.
        """
        if not self._running.value:
            return
        self._running.value = False
        # cascade stop
        try: self.q_raw.put_nowait(SENTINEL)
        except Full: pass
        try: self.q_proc.put_nowait(SENTINEL)
        except Full: pass

        if self.p_cap and self.p_cap.is_alive():
            self.p_cap.join(timeout=join_timeout)
            if self.p_cap.is_alive(): self.p_cap.terminate()
        if self.p_anon and self.p_anon.is_alive():
            self.p_anon.join(timeout=join_timeout)
            if self.p_anon.is_alive(): self.p_anon.terminate()
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=join_timeout)

        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        logger.info("[main] VideoStream stopped.")
