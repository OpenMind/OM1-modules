# -*- coding: utf-8 -*-
# om1_vlm/video/video_stream.py
import os, cv2, time, base64, inspect, asyncio, logging, platform, multiprocessing as mp
from queue import Full, Empty
import threading
from typing import Optional, Callable, List, Tuple

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
def proc_capture(out_q: mp.Queue, cam: str, res: Tuple[int, int], fps: int,
                 buffer_frames: int, verbose: bool = False):
    try:
        # 强制 V4L2，避免 GStreamer 报警
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if not cap.isOpened():
            # 兜底试几个常见设备
            for dev in ("/dev/video0", "/dev/video1", "/dev/video2"):
                cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
                if cap.isOpened():
                    cam = dev
                    break
        if not cap.isOpened():
            logger.error(f"[cap] cannot open {cam}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS,          fps)
        try:
            # MJPG 能大幅减轻 CPU 压力
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, buffer_frames))
        except Exception:
            pass

        if verbose:
            fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_i >> 8 * i) & 0xFF) for i in range(4)])
            rep = cap.get(cv2.CAP_PROP_FPS)
            print(f"[cap] using {cam} fourcc={fourcc} target_fps={fps} cap_prop_fps={rep:.1f}", flush=True)

        frame_time = 1.0 / max(1, fps)
        last = time.perf_counter()

        # 每秒统计一次采集 FPS
        cap_cnt, cap_t0 = 0, time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            ts = time.time()
            pkt = (ts, frame)

            try:
                out_q.put_nowait(pkt)
            except Full:
                # 丢弃一个旧帧让位
                try:
                    out_q.get_nowait()
                except Empty:
                    pass
                try:
                    out_q.put_nowait(pkt)
                except Full:
                    pass

            # 采集节流
            elapsed = time.perf_counter() - last
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            last = time.perf_counter()

            # 打点
            cap_cnt += 1
            if verbose and (time.perf_counter() - cap_t0) >= 1.0:
                print(f"[cap] fps={cap_cnt}", flush=True)
                cap_cnt, cap_t0 = 0, time.perf_counter()

    except KeyboardInterrupt:
        pass
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        if verbose:
            print("[cap] exit.", flush=True)


# ----------------------------
# Process B: Anonymize (TRT + pixelate)
# ----------------------------
def proc_anonymize(in_q: mp.Queue, out_q: mp.Queue,
                   scrfd_cfg: dict, blur_enabled: bool,
                   draw_boxes: bool, verbose: bool = False):
    anonymizer = None
    if blur_enabled and scrfd_cfg.get("engine_path"):
        anonymizer = _build_anonymizer(scrfd_cfg)

    # 统计
    anon_cnt, t0 = 0, time.perf_counter()
    sum_gpu_ms, sum_pix_ms = 0.0, 0.0

    try:
        while True:
            try:
                ts, frame = in_q.get(timeout=1.0)
            except Empty:
                continue
            if (ts, frame) == SENTINEL:
                break

            if blur_enabled and anonymizer is not None:
                t_pix0 = time.perf_counter()
                frame, dets, gpu_ms = anonymizer(frame)  # gpu_ms: TRT 推理时间
                # 像素化=总耗时-推理耗时
                pix_ms = (time.perf_counter() - t_pix0) * 1000.0 - (gpu_ms or 0.0)
                if pix_ms < 0:
                    pix_ms = 0.0

                if draw_boxes and dets is not None:
                    draw_dets(frame, dets)

                sum_gpu_ms += (gpu_ms or 0.0)
                sum_pix_ms += pix_ms

            # 输出
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

            # 打点
            anon_cnt += 1
            if verbose and (time.perf_counter() - t0) >= 1.0:
                avg_gpu = (sum_gpu_ms / anon_cnt) if anon_cnt else 0.0
                avg_pix = (sum_pix_ms / anon_cnt) if anon_cnt else 0.0
                print(f"[anon] fps={anon_cnt} avg_gpu_ms={avg_gpu:.1f} avg_pixelate_ms={avg_pix:.1f}", flush=True)
                anon_cnt, t0 = 0, time.perf_counter()
                sum_gpu_ms, sum_pix_ms = 0.0, 0.0
    finally:
        try:
            out_q.put_nowait(SENTINEL)
        except Full:
            pass
        if verbose:
            print("[anon] exit.", flush=True)


def _build_anonymizer(cfg: dict):
    class _Anon:
        def __init__(self, cfg):
            self.inf = TRTInfer(
                engine_path=cfg["engine_path"],
                input_name=cfg.get("input_name", "input.1"),
                size=cfg.get("size", 640),
                verbose=cfg.get("verbose", False),
            )
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
            dets, gpu_ms = self.inf.infer(frame_bgr)  # gpu_ms: 推理耗时(ms)
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


# ----------------------------
# Main-side wrapper
# ----------------------------
class VideoStream:
    def __init__(
        self,
        frame_callbacks: Optional[List[Callable[[str], None]]] = None,
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
        verbose: bool = False,
    ):
        self.fps = fps
        self.resolution = resolution
        self.encode_quality = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        self.verbose = bool(verbose)

        # self.frame_callbacks = frame_callbacks or []
        self._cb_lock = threading.Lock()
        if frame_callbacks is None:
            self.frame_callbacks = []
        elif callable(frame_callbacks):
            # allow passing a single function
            self.frame_callbacks = [frame_callbacks]
        else:
            self.frame_callbacks = list(frame_callbacks)

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(target=self._start_loop, daemon=True)
        self.loop_thread.start()

        self.cam = camera if camera is not None else _pick_camera()
        self.q_raw = mp.Queue(maxsize=queue_size_raw)
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
            verbose=self.verbose,
        )
        self.blur_enabled = bool(blur_enabled)
        self.draw_boxes = draw_boxes
        self.buffer_frames = buffer_frames

        self.p_cap: Optional[mp.Process] = None
        self.p_anon: Optional[mp.Process] = None
        self._drain_thread: Optional[threading.Thread] = None
        self._running = mp.Value("b", False)
    

    def register_frame_callback(self, cb: Callable[[str], None]) -> None:
        """Register an additional per-frame callback (receives base64 JPEG)."""
        if cb is None:
            return
        with self._cb_lock:
            if cb not in self.frame_callbacks:
                self.frame_callbacks.append(cb)

    def unregister_frame_callback(self, cb: Callable[[str], None]) -> None:
        """Unregister a previously added callback (no error if missing)."""
        with self._cb_lock:
            try:
                self.frame_callbacks.remove(cb)
            except ValueError:
                pass


    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def start(self):
        if self._running.value:
            return
        self._running.value = True
        try:
            mp.set_start_method("spawn", force=False)
        except RuntimeError:
            pass

        self.p_cap = mp.Process(
            target=proc_capture,
            args=(self.q_raw, self.cam, self.resolution, self.fps, self.buffer_frames, self.verbose),
            daemon=True,
        )
        self.p_anon = mp.Process(
            target=proc_anonymize,
            args=(self.q_raw, self.q_proc, self.scrfd_cfg, self.blur_enabled, self.draw_boxes, self.verbose),
            daemon=True,
        )
        self.p_cap.start()
        self.p_anon.start()

        self._drain_thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._drain_thread.start()
        logger.info("[main] VideoStream started.")

    def _dispatch(self, b64_jpeg: str):
        with self._cb_lock:
            callbacks = tuple(self.frame_callbacks)  # snapshot to avoid mutation during iteration
        for cb in callbacks:
            if inspect.iscoroutinefunction(cb):
                asyncio.run_coroutine_threadsafe(cb(b64_jpeg), self.loop)
            else:
                cb(b64_jpeg)


    def _drain_loop(self):
        """
        Main-process drain loop.
        - Pull processed frames
        - JPEG encode + base64
        - Dispatch to callbacks
        - Pace to ~fps
        """
        frame_time = 1.0 / max(1, self.fps)
        last = time.perf_counter()

        # 打点
        read_cnt, read_t0 = 0, time.perf_counter()
        sum_enc_ms, sum_b64_ms = 0.0, 0.0

        try:
            while self._running.value:
                try:
                    ts, frame = self.q_proc.get(timeout=0.5)
                except Empty:
                    continue
                if (ts, frame) == SENTINEL:
                    break

                t0 = time.perf_counter()
                ok, buf = cv2.imencode(".jpg", frame, self.encode_quality)
                enc_ms = (time.perf_counter() - t0) * 1000.0

                if ok:
                    t1 = time.perf_counter()
                    self._dispatch(base64.b64encode(buf).decode("utf-8"))
                    b64_ms = (time.perf_counter() - t1) * 1000.0
                    sum_enc_ms += enc_ms
                    sum_b64_ms += b64_ms

                # 统一节流
                elapsed = time.perf_counter() - last
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
                last = time.perf_counter()

                # 打点
                read_cnt += 1
                if self.verbose and (time.perf_counter() - read_t0) >= 1.0:
                    avg_enc = (sum_enc_ms / read_cnt) if read_cnt else 0.0
                    avg_b64 = (sum_b64_ms / read_cnt) if read_cnt else 0.0
                    print(f"[main] read_fps={read_cnt} avg_jpeg_ms={avg_enc:.1f} avg_b64_cb_ms={avg_b64:.1f}", flush=True)
                    read_cnt, read_t0 = 0, time.perf_counter()
                    sum_enc_ms, sum_b64_ms = 0.0, 0.0
        finally:
            if self.verbose:
                print("[main] drain exit.", flush=True)

    def stop(self, join_timeout: float = 1.0):
        if not self._running.value:
            return
        self._running.value = False
        try:
            self.q_raw.put_nowait(SENTINEL)
        except Full:
            pass
        try:
            self.q_proc.put_nowait(SENTINEL)
        except Full:
            pass

        if self.p_cap and self.p_cap.is_alive():
            self.p_cap.join(timeout=join_timeout)
            if self.p_cap.is_alive():
                self.p_cap.terminate()
        if self.p_anon and self.p_anon.is_alive():
            self.p_anon.join(timeout=join_timeout)
            if self.p_anon.is_alive():
                self.p_anon.terminate()
        if self._drain_thread and self._drain_thread.is_alive():
            self._drain_thread.join(timeout=join_timeout)

        try:
            self.loop.call_soon_threadsafe(self.loop.stop)
        except Exception:
            pass
        logger.info("[main] VideoStream stopped.")
