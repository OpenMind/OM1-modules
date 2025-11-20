#!/usr/bin/env python3
"""
Real-time face anonymization + recognition + RTSP + HTTP control.

This program captures video from a V4L2 camera, detects faces (SCRFD, TensorRT),
optionally recognizes them (ArcFace, TensorRT) using an on-disk gallery managed by
`GalleryManager`, applies privacy pixelation (all / known / unknown), and publishes
the processed video to RTSP (local and/or remote). A lightweight HTTP control plane
lets you query presence (/who), tweak runtime config (/config), and update the gallery
(/gallery/refresh, /gallery/add_aligned, /gallery/add_raw, /selfie), all while the
main video loop keeps running.

- Detection: SCRFD (TensorRT)
- Recognition: ArcFace (TensorRT) via GalleryManager (incremental refresh)
- Privacy: pixelation (all / known / unknown)
- Streaming: RTSP (local; remote relay handled by MediaMTX/FFmpeg)
- HTTP: /who, /config (get/set), /gallery/refresh, /gallery/add_aligned,
        /gallery/add_raw, /selfie, /ping, /ts

Usage examples
--------------
# Minimal (local RTSP only)
python -m om1_vlm.anonymizationSys.face_recog_stream.run \
  --device /dev/video0 --width 1280 --height 720 --fps 30 \
  --detection --recognition --blur --blur-mode all \
  --draw-boxes --draw-names --show-fps \
  --http-host 0.0.0.0 --http-port 6791

# With remote RTSP relay
python -m om1_vlm.anonymizationSys.face_recog_stream.run \
  --device /dev/video0 --width 1280 --height 720 --fps 30 \
  --detection --recognition --blur --blur-mode unknown \
  --draw-boxes --draw-names --show-fps \
  --remote-rtsp "rtsp://api-video-ingest.openmind.org:8554/<stream>?api_key=<KEY>" \
  --http-host 0.0.0.0 --http-port 6791

# Use teach_face.sh for simple
# Query presence (who's been seen in last 2s)
curl -s -X POST http://127.0.0.1:6791/who -d '{"recent_sec":2}' -H 'Content-Type: application/json'

# Read config / change config
curl -s -X POST http://127.0.0.1:6791/config -d '{"get":true}' -H 'Content-Type: application/json'
curl -s -X POST http://127.0.0.1:6791/config -d '{"set":{"blur_mode":"unknown","sim_thr":0.4}}' \
  -H 'Content-Type: application/json'

# Refresh gallery (process new RAW → ALIGNED, embed, update means)
curl -s -X POST http://127.0.0.1:6791/gallery/refresh -d '{}' -H 'Content-Type: application/json'

# Add aligned snapshot from disk (112x112) or from base64
curl -s -X POST http://127.0.0.1:6791/gallery/add_aligned \
  -d '{"id":"alice","image_path":"/path/to/aligned_112.jpg"}' -H 'Content-Type: application/json'

# Add raw photo (any size) then refresh+embed
curl -s -X POST http://127.0.0.1:6791/gallery/add_raw \
  -d '{"id":"alice","image_path":"/path/to/photo.jpg"}' -H 'Content-Type: application/json'

# Selfie enrollment from the last clean frame (must have exactly 1 face)
curl -s -X POST http://127.0.0.1:6791/selfie -d '{"id":"alice"}' -H 'Content-Type: application/json'

# Delete one person identity from gallery
curl -sS -X POST 'http://127.0.0.1:6793/gallery/delete' \
  -H 'Content-Type: application/json' \
  -d '{"id":"boyuan"}' | jq .

# Delete multiple person identity from gallery  
curl -sS -X POST http://127.0.0.1:6793/gallery/delete \
  -H 'Content-Type: application/json' \
  -d '{"ids":["wendy","boyuan"]}' | jq .

# Check and list gallery identies  
curl -sS -X POST http://127.0.0.1:6793/gallery/identities -H 'Content-Type: application/json' -d '{}' | jq .


Check teach_face.sh for quick usage
-------------------------
The helper script `teach_face.sh` is a thin wrapper around the HTTP API above
(useful from terminals or automation). Examples:

# default recent_sec = 2
./teach_face.sh who

# 1-second window
./teach_face.sh who 1

# change a setting (e.g., detector confidence)
./teach_face.sh config set conf=0.2

# check current config
./teach_face.sh config get

# take a selfie for “wendy”
./teach_face.sh selfie wendy

# upload an existing image to raw, then refresh/align/embed
./teach_face.sh upload wendy /abs/path/to/image.jpg

# delete multiple person identity from gallery
./teach_face.sh delete wendy alice bob

Notes
-----
- The script targets the same HTTP service this process starts (e.g. http://127.0.0.1:6791).
  If your script has host/port variables, adjust them to match `--http-host` and `--http-port`.
- All GPU/model work is executed on the **main thread** via a job-queue drain at the
  start of each frame, ensuring TensorRT engines are used from a single thread.
- HTTP endpoints are responsive but cannot starve the frame loop: at most a few jobs
  are executed per frame (tunable).
- `--perf-mode` can skip recognition when frame time exceeds budget.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from queue import Empty, Queue
from typing import Dict, List, Optional

import cv2
import numpy as np

from om1_utils.http import Server  # lightweight HTTP server

from .arcface import TRTArcFace, warp_face_by_5p
from .camera_reader import CameraReader
from .draw import draw_overlays
from .gallery import GalleryManager
from .http_api import HttpAPI
from .rtsp_video_writer import RTSPVideoStreamWriter
from .scrfd import TRTSCRFD
from .utils import infer_arc_batched, pick_topk_indices
from .who_tracker import WhoTracker

logger = logging.getLogger(__name__)


# ---------------------------- Small shared states -------------------------- #
class _GalState:
    """Mutable in-memory gallery state shared with the main loop and HTTP.

    Attributes
    ----------
    gal_feats : np.ndarray | None
        Stacked, L2-normalized per-identity mean vectors (N_id × dim) used for
        cosine similarity at runtime. May be None if the gallery is empty.
    gal_labels : list[str]
        Identity labels aligned with `gal_feats` rows.
    """

    def __init__(self):
        self.gal_feats: Optional[np.ndarray] = None
        self.gal_labels: List[str] = []


class _FrameState:
    """Holds the last clean frame and detections for HTTP-driven enrollment.

    Captured **before** any overlays/blur, so `/selfie` can save a pristine image
    without extra inference.

    Attributes
    ----------
    frame_bgr : np.ndarray | None
        Last raw frame (BGR). None until a frame is seen.
    dets : np.ndarray | None
        Detection array of shape (N, 5) with [x1, y1, x2, y2, score], or None.
    kpss : np.ndarray | None
        Optional 5-point landmarks per detection, shape (N, 5, 2), or None.
    """

    def __init__(self):
        self.frame_bgr: Optional[np.ndarray] = None
        self.dets: Optional[np.ndarray] = None
        self.kpss: Optional[np.ndarray] = None


# --------------------------------- Main ----------------------------------- #
def main() -> None:
    """Entry point: wire up models, gallery, HTTP, streaming, and run the loop.

    Flow
    ----
    1) Parse CLI args and resolve default paths.
    2) Load TensorRT engines (SCRFD detector, optional ArcFace).
    3) Build/refresh gallery via `GalleryManager` and cache identity means.
    4) Open camera (`CameraReader`) and start RTSP writer.
    5) Initialize shared states, locks, and an HTTP `HttpAPI` bound to a tiny
       `Server` that runs on its own thread.
    6) Start the main loop:
       - Drain up to a few queued jobs (from HTTP) on the **main thread**.
       - Read a frame; detect faces; cache a clean copy for `/selfie`.
       - If recognition is enabled and within budget, embed selected faces and
         match to identity means.
       - Update the WhoTracker; apply pixelation; draw overlays; stream frame.
       - Keep simple pacing to meet target FPS; log periodic stats.

    Side Effects
    ------------
    - Binds an HTTP server on `--http-host:--http-port`.
    - Publishes RTSP to `--local-rtsp` and optionally mirrors to `--remote-rtsp`.
    - Responds to SIGINT for graceful shutdown.

    Returns
    -------
    None
    """
    logger.info("Starting realtime_stream...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    platform = os.environ.get("OM1_PLATFORM","").lower()
    if platform == "thor":
        scrfd_name = "thor_buffalo_m_w600k_r50.engine"
        arc_name = "thor_buffalo_m_w600k_r50.engine"
    elif platfrom == "orin":
        scrfd_name = "orin_scrfd_2.5g_bnkps_shape640x640.engine"
        arc_name = "orin_buffalo_m_w600k_r50.engine"
    else:
        scrfd_name = "thor_buffalo_m_w600k_r50.engine"
        arc_name = "thor_buffalo_m_w600k_r50.engine"
        
    default_scrfd_engine = os.path.join(
        models_dir, scrfd_name
    )
    default_arc_engine = os.path.join(models_dir, arc_name)
    default_gallery = os.path.join(script_dir, "..", "gallery")

    ap = argparse.ArgumentParser(
        "Jetson real-time detection + recognition + blur + RTSP + HTTP"
    )

    # Engines
    ap.add_argument(
        "--scrfd-engine",
        default=default_scrfd_engine,
        help="Path to SCRFD TensorRT engine (.engine).",
    )
    ap.add_argument(
        "--scrfd-input", default=None, help="SCRFD input tensor name (optional)."
    )
    ap.add_argument(
        "--size", type=int, default=640, help="SCRFD input size (square; e.g., 640)."
    )
    ap.add_argument(
        "--arc-engine",
        default=default_arc_engine,
        help="Path to ArcFace TensorRT engine (.engine).",
    )
    ap.add_argument(
        "--arc-input", default=None, help="ArcFace input tensor name (optional)."
    )
    ap.add_argument(
        "--gallery",
        dest="gallery_dir",
        default=default_gallery,
        help="Gallery root (subfolders per identity). Required if --recognition.",
    )
    ap.add_argument(
        "--embeds-dir",
        dest="embeds_dir",
        default=None,
        help="Embeddings root. Default is sibling of gallery, e.g. /path/to/embeds for gallery /path/to/gallery.",
    )

    # Features
    ap.add_argument(
        "--detection", action="store_true", default=True, help="Enable face detection."
    )
    ap.add_argument(
        "--recognition",
        action="store_true",
        default=True,
        help="Enable face recognition (uses gallery).",
    )
    ap.add_argument(
        "--blur",
        action="store_true",
        default=True,
        help="Enable pixelation blur on faces.",
    )
    ap.add_argument(
        "--blur-mode",
        choices=["all", "known", "unknown"],
        default="all",
        help="Which faces to blur.",
    )
    ap.add_argument(
        "--pixel-blocks",
        type=int,
        default=8,
        help="Pixelation blocks on the short side.",
    )
    ap.add_argument(
        "--pixel-margin",
        type=float,
        default=0.25,
        help="Expand box before blur (fraction of size).",
    )
    ap.add_argument(
        "--pixel-noise",
        type=float,
        default=0.0,
        help="Add Gaussian noise to blurred area (0=off).",
    )
    ap.add_argument("--draw-boxes", action="store_true", help="Draw detection boxes.")
    ap.add_argument(
        "--draw-names", action="store_true", help="Draw recognized names/scores."
    )
    ap.add_argument(
        "--show-fps", action="store_true", help="Overlay per-frame time and FPS."
    )
    ap.add_argument(
        "--recog-topk",
        type=int,
        default=4,
        help="Max faces per frame to send to ArcFace (<= ArcFace engine max batch).",
    )
    ap.add_argument(
        "--crowd-thr",
        type=int,
        default=12,
        help="If detected faces exceed this, skip recognition (still blur/draw).",
    )

    # Thresholds
    ap.add_argument(
        "--conf", type=float, default=0.5, help="Detection confidence threshold."
    )
    ap.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold.")
    ap.add_argument(
        "--sim-thr",
        type=float,
        default=0.35,
        help="Recognition cosine similarity threshold.",
    )
    ap.add_argument(
        "--max-num", type=int, default=0, help="Max faces to keep (0 = no limit)."
    )

    # Inputs
    ap.add_argument(
        "--device", default="/dev/video0", help="V4L2 device (e.g., /dev/video0)"
    )
    ap.add_argument("--width", type=int, default=640, help="Capture width.")
    ap.add_argument("--height", type=int, default=480, help="Capture height.")
    ap.add_argument("--fps", type=int, default=30, help="Capture frame rate.")

    # Outputs
    ap.add_argument(
        "--local-rtsp",
        default="rtsp://localhost:8554/top_camera",
        help="RTSP URL to publish (e.g. rtsp://host:8554/top_camera).",
    )
    ap.add_argument(
        "--remote-rtsp",
        help="Remote RTSP URL to relay (e.g. rtsp://host:8554/top_camera).",
    )

    # UI / perf
    ap.add_argument(
        "--no-window", action="store_true", help="Disable display window (headless)."
    )
    ap.add_argument("--print-every", type=int, default=30, help="Log every N frames.")
    ap.add_argument(
        "--perf-mode",
        action="store_true",
        help="Enable performance mode (skip recognition when processing is slow).",
    )

    # HTTP control
    ap.add_argument("--http-host", default="127.0.0.1", help="HTTP bind host.")
    ap.add_argument("--http-port", type=int, default=6793, help="HTTP bind port.")
    ap.add_argument(
        "--http-lookback-sec",
        type=float,
        default=10.0,
        help="Default lookback window for /who.",
    )

    args = ap.parse_args()

    # Resolve default embeds_dir if not provided
    if args.embeds_dir is None:
        parent = os.path.dirname(os.path.abspath(args.gallery_dir))
        args.embeds_dir = os.path.join(parent, "embeds")

    # Sanity checks
    if not os.path.exists(args.scrfd_engine):
        raise SystemExit(f"SCRFD engine not found: {args.scrfd_engine}")
    if args.recognition:
        if not os.path.exists(args.arc_engine):
            raise SystemExit(f"ArcFace engine not found: {args.arc_engine}")
        if not os.path.exists(args.gallery_dir):
            raise SystemExit(f"Gallery directory not found: {args.gallery_dir}")
        os.makedirs(args.embeds_dir, exist_ok=True)

    # Engines
    scrfd = TRTSCRFD(args.scrfd_engine, input_name=args.scrfd_input, size=args.size)
    scrfd.nms_thresh = args.nms

    arc: Optional[TRTArcFace] = None
    if args.recognition:
        arc = TRTArcFace(args.arc_engine, input_name=args.arc_input)

    # Safe ArcFace batch (avoid profile errors)
    arc_max_bs = 4
    if args.recognition and arc is not None and hasattr(arc, "max_batch"):
        try:
            arc_max_bs = max(1, int(arc.max_batch))  # type: ignore[attr-defined]
        except Exception:
            arc_max_bs = 4

    # Gallery state + lock
    gal_state = _GalState()
    gal_lock = threading.Lock()
    gm: Optional[GalleryManager] = None

    if args.recognition:
        gm = GalleryManager(
            gallery_dir=args.gallery_dir,
            embeds_dir=args.embeds_dir,
            arc=arc,  # type: ignore[arg-type]
            scrfd=scrfd,
            det_conf=args.conf,
        )
        t0b = time.time()
        aligned_added, vec_added = gm.refresh(process_raw=True)
        feats, id_labels = gm.get_identity_means()
        with gal_lock:
            gal_state.gal_feats, gal_state.gal_labels = feats, id_labels
        logger.info(
            "Gallery ready: identities=%d (aligned+%d, vectors+%d) time=%.2fs",
            len(id_labels),
            aligned_added,
            vec_added,
            time.time() - t0b,
        )

    # Capture
    cap = CameraReader(args.device, args.width, args.height, args.fps)
    if not cap.is_opened():
        raise SystemExit(f"Failed to open camera {args.device}")

    # Streamer (processed frames)
    rstp_writer = RTSPVideoStreamWriter(
        cap.width,
        cap.height,
        cap.fps,
        args.local_rtsp,
        args.remote_rtsp,
    )
    logger.info(
        "Publishing RTSP: local=%s%s",
        args.local_rtsp,
        f"  remote={args.remote_rtsp}" if args.remote_rtsp else "",
    )

    # Who tracking
    who = WhoTracker(lookback_sec=args.http_lookback_sec)

    # Runtime config (HTTP-settable)
    cfg_lock = threading.Lock()
    cfg: Dict[str, object] = {
        "blur": bool(args.blur),
        "blur_mode": str(args.blur_mode),
        "sim_thr": float(args.sim_thr),
        "recog_topk": int(args.recog_topk),
        "crowd_thr": int(args.crowd_thr),
        "perf_mode": bool(args.perf_mode),
        "show_fps": bool(args.show_fps),
        "conf": float(args.conf),
        "nms": float(args.nms),
        "max_num": int(args.max_num),
        "pixel_blocks": int(args.pixel_blocks),
        "pixel_margin": float(args.pixel_margin),
        "pixel_noise": float(args.pixel_noise),
    }

    # Last clean frame & detections for /selfie
    frame_state = _FrameState()
    frame_lock = threading.Lock()

    # ----------- Job queue so CUDA work runs on the main thread only -------- #
    class Job:
        """A unit of work to be executed on the main thread.

        Parameters
        ----------
        fn : Callable[[], Any]
            The function to execute (may perform CUDA/TensorRT work).

        Attributes
        ----------
        done : threading.Event
            Signals completion (set by `run()`).
        result : Any
            Return value of `fn()` if it succeeded.
        exc : Exception | None
            Captured exception if `fn()` raised.
        """

        def __init__(self, fn):
            self.fn = fn
            self.done = threading.Event()
            self.result = None
            self.exc: Optional[Exception] = None

        def run(self):
            """Execute the job function, capturing result or exception.

            Sets `done` in a finally-block so waiting threads are always released.
            """
            try:
                self.result = self.fn()
            except Exception as e:
                self.exc = e
            finally:
                self.done.set()

    jobs: "Queue[Job]" = Queue()

    def run_job_sync(fn):
        """Enqueue work for the main thread and block until it completes.

        This is called by HTTP handlers to offload GPU/model operations to the
        main loop. It ensures TensorRT engines are only driven from one thread.

        Parameters
        ----------
        fn : Callable[[], Any]
            Function to be executed on the main thread.

        Returns
        -------
        Any
            The return value of `fn()`.

        Raises
        ------
        Exception
            Re-raises any exception raised by `fn()`.
        """
        jb = Job(fn)
        jobs.put(jb)
        jb.done.wait()
        if jb.exc:
            raise jb.exc
        return jb.result

    http_api = HttpAPI(
        who=who,
        scrfd=scrfd,
        gm=gm,
        gallery_dir=args.gallery_dir,
        gal_state=gal_state,
        gal_lock=gal_lock,
        cfg=cfg,
        cfg_lock=cfg_lock,
        frame_state=frame_state,
        frame_lock=frame_lock,
        run_job_sync=run_job_sync,
        logger=logger,
    )

    # Run the server here
    http = Server(host=args.http_host, port=args.http_port, timeout=15)

    http.register_message_callback(http_api._handle)

    http.start()
    logger.info("HTTP listening on http://%s:%d", args.http_host, args.http_port)

    # Graceful shutdown
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        logger.info("SIGINT received, shutting down....")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.perf_counter()
    total = 0
    ema_ms: Optional[float] = None
    target_frame_time = 1.0 / max(1, args.fps)
    next_frame_time = time.perf_counter()

    try:
        while running:
            # Execute any queued CUDA jobs (from HTTP requests) on the main thread.
            # We drain only a few per frame to keep the UI responsive without starving FPS.
            for _ in range(4):  # drain a few per frame to keep UI snappy
                try:
                    jb = jobs.get_nowait()
                except Empty:
                    break
                jb.run()

            frame = cap.read_frame()
            if frame is None:
                logger.warning("[warn] Frame is None ")
                time.sleep(0.02)
                continue

            # Snapshot config
            with cfg_lock:
                do_blur = bool(cfg["blur"])
                blur_mode = str(cfg["blur_mode"])
                sim_thr = float(cfg["sim_thr"])
                recog_topk = int(cfg["recog_topk"])
                crowd_thr = int(cfg["crowd_thr"])
                perf_mode = bool(cfg["perf_mode"])
                show_fps = bool(cfg["show_fps"])
                det_conf = float(cfg["conf"])
                max_num = int(cfg["max_num"])
                # pixel_blocks = int(cfg["pixel_blocks"])
                # pixel_margin = float(cfg["pixel_margin"])
                # pixel_noise = float(cfg["pixel_noise"])

            total += 1
            t_start = time.perf_counter()

            # Detection
            dets = np.zeros((0, 5), np.float32)
            kpss: Optional[np.ndarray] = None
            try:
                dets, kpss = scrfd.detect(frame, conf=det_conf, max_num=max_num)
            except Exception as e:
                logger.warning("[warn] detection failed this frame: %s", e)
                dets, kpss = np.zeros((0, 5), np.float32), None

            # Save CLEAN copy + dets/kpss for /selfie BEFORE any drawing/blur
            with frame_lock:
                frame_state.frame_bgr = frame.copy()
                frame_state.dets = None if dets is None else dets.copy()
                frame_state.kpss = None if kpss is None else kpss.copy()

            # Recognition (Top-K with perf guard)
            names: List[Optional[str]] = []
            known_mask: List[bool] = []

            skip_recognition = (
                perf_mode
                and ema_ms is not None
                and ema_ms > (target_frame_time * 1000.0 * 0.8)
            )  # ema in ms vs 80% of frame budget

            with gal_lock:
                have_gallery = (
                    gal_state.gal_feats is not None and gal_state.gal_feats.size > 0
                )

            do_recog = (
                args.recognition
                and (not skip_recognition)
                and dets is not None
                and dets.shape[0] > 0
                and have_gallery
            )

            if do_recog and (dets.shape[0] <= crowd_thr):
                try:
                    Hf, Wf = frame.shape[:2]
                    n_det = dets.shape[0]
                    names_full: List[Optional[str]] = ["unknown"] * n_det
                    known_mask = [False] * n_det

                    sel_idx = (
                        np.arange(n_det, dtype=np.int32)
                        if n_det <= recog_topk
                        else pick_topk_indices(dets, topk=recog_topk, H=Hf, W=Wf)
                    )

                    crops: List[np.ndarray] = []
                    backmap: List[int] = []
                    for i_sel in sel_idx:
                        try:
                            if kpss is not None:
                                crop = warp_face_by_5p(frame, kpss[i_sel], 112)
                            else:
                                x1, y1, x2, y2, _ = dets[i_sel].astype(int)
                                face = frame[
                                    max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)
                                ]
                                crop = (
                                    cv2.resize(face, (112, 112))
                                    if face.size > 0
                                    else None
                                )
                            if crop is not None and crop.size:
                                crops.append(crop)
                                backmap.append(int(i_sel))
                        except Exception:
                            pass

                    if crops and arc is not None:
                        # Respect engine profile
                        max_bs = min(arc_max_bs, max(1, recog_topk))
                        feats_list = []
                        for i in range(0, len(crops), max_bs):
                            sub = crops[i : i + max_bs]
                            vecs = infer_arc_batched(arc, sub, max_bs=max_bs)  # type: ignore[arg-type]
                            vecs = vecs / (
                                np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
                            )
                            feats_list.append(vecs.astype(np.float32))
                        feats = (
                            np.concatenate(feats_list, axis=0)
                            if len(feats_list) > 1
                            else feats_list[0]
                        )

                        with gal_lock:
                            S = feats @ gal_state.gal_feats.T  # type: ignore[union-attr]
                            best_j = np.argmax(S, axis=1)
                            best_v = S[np.arange(S.shape[0]), best_j]
                            labels_ref = gal_state.gal_labels

                        for pos, (vv, jj) in enumerate(zip(best_v, best_j)):
                            det_i = backmap[pos]
                            if float(vv) >= float(sim_thr):
                                label = labels_ref[int(jj)]
                                names_full[det_i] = f"{label} ({float(vv):.2f})"
                                known_mask[det_i] = True
                            else:
                                names_full[det_i] = "unknown"

                    names = names_full
                except Exception as e:
                    logger.warning("[warn] recognition failed this frame: %s", e)
                    names = [None] * dets.shape[0]
                    known_mask = [False] * dets.shape[0]
            else:
                n = 0 if dets is None else int(dets.shape[0])
                names = ["unknown"] * n
                known_mask = [False] * n
                # if args.recognition and dets is not None and dets.shape[0] > crowd_thr:
                #     names = ["unknown"] * int(dets.shape[0])

            # Who-tracker (strip score suffix)
            def strip_score(nm: Optional[str]) -> Optional[str]:
                if nm is None:
                    return None
                if nm == "unknown":
                    return "unknown"
                p = nm.find(" (")
                return nm[:p] if p > 0 else nm

            who.update_now([strip_score(nm) for nm in names])

            # Blur
            if do_blur and dets is not None and dets.shape[0] > 0:
                from .pixelate import expand_clip, pixelate_roi

                Hf, Wf = frame.shape[:2]
                for i, (x1, y1, x2, y2, _) in enumerate(dets):
                    apply = True
                    if blur_mode != "all" and args.recognition:
                        is_known = known_mask[i] if i < len(known_mask) else False
                        apply = (blur_mode == "known" and is_known) or (
                            blur_mode == "unknown" and not is_known
                        )
                    if apply:
                        x1e, y1e, x2e, y2e = expand_clip(
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            float(cfg["pixel_margin"]),
                            Wf,
                            Hf,
                        )
                        pixelate_roi(
                            frame,
                            x1e,
                            y1e,
                            x2e,
                            y2e,
                            blocks_on_short=int(cfg["pixel_blocks"]),
                            noise_sigma=float(cfg["pixel_noise"]),
                        )

            # Overlays
            if args.draw_boxes or args.draw_names:
                frame = draw_overlays(
                    frame,
                    dets,
                    names,
                    kpss,
                    draw_boxes=args.draw_boxes,
                    draw_names=args.draw_names,
                )

            # Stats
            dt_ms = (time.perf_counter() - t_start) * 1000.0
            ema_ms = dt_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * dt_ms)
            if show_fps:
                sec = max(1e-9, time.perf_counter() - t0)
                fps_now = total / sec
                overlay = f"{dt_ms:.1f} ms (EMA {ema_ms:.1f}) | {fps_now:.1f} FPS | faces {0 if dets is None else dets.shape[0]}"
                cv2.putText(
                    frame,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (40, 220, 40),
                    2,
                    cv2.LINE_AA,
                )

            if rstp_writer is not None:
                rstp_writer.write_frame(frame)

            # Optional local preview
            if not args.no_window:
                cv2.imshow("Face Anonymizer", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            next_frame_time += target_frame_time
            current_time = time.perf_counter()
            sleep_time = next_frame_time - current_time

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = current_time

            if total % max(1, args.print_every) == 0:
                sec = max(1e-9, time.perf_counter() - t0)
                fps_now = total / sec
                logger.info(
                    "[%05d] frame=%.2f ms  EMA=%.2f  FPS=%.1f  faces=%d",
                    total,
                    dt_ms,
                    ema_ms or 0.0,
                    fps_now,
                    0 if dets is None else dets.shape[0],
                )

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            http.stop()
        except Exception:
            pass
        try:
            rstp_writer.stop()
        except Exception:
            pass
        if not args.no_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
