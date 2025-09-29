#!/usr/bin/env python3
"""
Usage examples
--------------

# Basic usage with default model paths (engines and gallery auto-detected):
python -m om1_vlm.anonymizationSys.face_recog_stream.run --device /dev/video0 --width 1280 --height 720 --fps 30 --detection --recognition --blur --blur-mode all --draw-boxes --draw-names --show-fps --recog-topk 8 --crowd-thr 12 --remote-rtsp "rtsp://api-video-ingest.openmind.org:8554/<API_KEY_ID>?api_key=<API_KEY>"

# Local mediamtx (RTMP), no preview window (uses default model paths):
  python -m om1_vlm.anonymizationSys.face_recog_stream.run  \
    --device /dev/video0 \
    --width 1280 --height 720 --fps 30 \
    --detection --recognition --blur --blur-mode all \
    --draw-boxes --draw-names --show-fps \
    --recog-topk 8 --crowd-thr 12 \
    --no-window

# Custom paths example:
  python -m om1_vlm.anonymizationSys.face_recog_stream.run  \
    --scrfd-engine "/path/to/scrfd_2.5g_bnkps_shape640x640.engine" \
    --arc-engine   "/path/to/buffalo_m_w600k_r50.engine" \
    --gallery      "/path/to/gallery" \
    --device /dev/video0 \
    --width 1280 --height 720 --fps 30 \
    --detection --recognition --blur --blur-mode all \
    --draw-boxes --draw-names --show-fps \
    --recog-topk 8 --crowd-thr 12 \
    --no-window
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from typing import List, Optional

import cv2
import numpy as np
import pycuda.autoinit  # noqa: F401

from .arcface import TRTArcFace, warp_face_by_5p
from .camera_reader import CameraReader
from .draw import draw_overlays
from .gallery import build_gallery_embeddings
from .rtsp_video_writer import RTSPVideoStreamWriter
from .scrfd import TRTSCRFD
from .utils import infer_arc_batched, pick_topk_indices

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Run real-time face detection/recognition/blur and optional RTMP/file output.

    This script:
      1) Opens a camera (OpenCV V4L2 or GStreamer).
      2) Runs SCRFD detection (TensorRT).
      3) Optionally runs ArcFace recognition against a gallery (TensorRT).
      4) Optionally pixelates faces (known/unknown/all).
      5) Draws boxes/names and optionally streams to RTMP and/or records to MP4.
    """
    logging.info("Starting realtime_stream...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "..", "models")
    default_scrfd_engine = os.path.join(
        models_dir, "scrfd_2.5g_bnkps_shape640x640.engine"
    )
    default_arc_engine = os.path.join(models_dir, "buffalo_m_w600k_r50.engine")
    default_gallery = os.path.join(script_dir, "..", "gallery")

    ap = argparse.ArgumentParser(
        "Jetson real-time detection + recognition + blur + streaming"
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
        default=default_gallery,
        help="Gallery root (subfolders per identity). Required if --recognition.",
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
        help="Max faces per frame to send to ArcFace (<= your ArcFace engine's max batch).",
    )
    ap.add_argument(
        "--crowd-thr",
        type=int,
        default=12,
        help="If detected faces in a frame exceed this threshold, skip recognition (still blur/draw).",
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
        "--device",
        default="/dev/video0",
        help="V4L2 device (e.g., /dev/video0)",
    )
    ap.add_argument("--width", type=int, default=640, help="Capture width.")
    ap.add_argument("--height", type=int, default=480, help="Capture height.")
    ap.add_argument("--fps", type=int, default=30, help="Capture frame rate.")

    # Outputs
    ap.add_argument(
        "--local-rtsp",
        default="rtsp://localhost:8554/live",
        help="RTSP URL to publish (e.g. rtsp://host:8554/stream).",
    )
    ap.add_argument(
        "--remote-rtsp", help="Remote RTSP URL to relay (e.g. rtsp://host:8554/stream)."
    )
    ap.add_argument(
        "--rtsp-mic-device",
        default="hw:3,0",
        help="Audio capture device for RTSP (e.g. hw:3,0).",
    )
    ap.add_argument(
        "--rtsp-mic-ac", type=int, default=2, help="Audio channels for RTSP (e.g. 2)."
    )
    ap.add_argument(
        "--no-window", action="store_true", help="Disable display window (headless)."
    )
    ap.add_argument("--print-every", type=int, default=30, help="Log every N frames.")
    ap.add_argument(
        "--perf-mode",
        action="store_true",
        help="Enable performance mode (skip recognition when processing is slow).",
    )

    args = ap.parse_args()

    if not os.path.exists(args.scrfd_engine):
        raise SystemExit(f"SCRFD engine not found: {args.scrfd_engine}")

    if args.recognition:
        if not os.path.exists(args.arc_engine):
            raise SystemExit(f"ArcFace engine not found: {args.arc_engine}")
        if not os.path.exists(args.gallery):
            raise SystemExit(f"Gallery directory not found: {args.gallery}")

    # Create engines
    scrfd = TRTSCRFD(args.scrfd_engine, input_name=args.scrfd_input, size=args.size)
    scrfd.nms_thresh = args.nms

    arc: Optional[TRTArcFace] = None
    gal_feats: Optional[np.ndarray] = None
    gal_labels: List[str] = []
    if args.recognition:
        arc = TRTArcFace(args.arc_engine, input_name=args.arc_input)
        logging.info(">> Building gallery embeddings…")
        t0_build = time.time()
        try:
            gal_feats, gal_labels = build_gallery_embeddings(
                args.gallery, scrfd, arc, det_conf=args.conf
            )
            logging.info(
                f"   identities={len(gal_labels)}  time={time.time() - t0_build:.2f}s"
            )
        except Exception as e:
            logging.warning(
                f"[warn] build_gallery_embeddings failed: {e}. Recognition will be disabled."
            )
            args.recognition = False
            arc = None
            gal_feats, gal_labels = None, []

    # Open capture
    cap = CameraReader(args.device, args.width, args.height, args.fps)
    if cap.is_opened() is False:
        raise SystemExit(f"Failed to open camera {args.device}")

    local_rtsp = args.local_rtsp if args.local_rtsp else None
    remote_rtsp = args.remote_rtsp if args.remote_rtsp else None

    rstp_writer = RTSPVideoStreamWriter(
        cap.width,
        cap.height,
        cap.fps,
        local_rtsp,
        remote_rtsp,
        args.rtsp_mic_device,
        args.rtsp_mic_ac,
    )
    logging.info(
        f"Start streaming to RTSP server at local {local_rtsp} "
        + (f"and remote {remote_rtsp}" if remote_rtsp else "")
    )

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        logging.info("SIGINT received, shutting down....")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.perf_counter()
    total = 0
    ema_ms: Optional[float] = None

    target_frame_time = 1.0 / args.fps
    last_frame_time = time.perf_counter()

    try:
        while running:
            frame = cap.read_frame()
            if frame is None:
                logging.warning("Frame is None ")
                time.sleep(0.1)
                continue

            total += 1
            start = time.perf_counter()

            dets = np.zeros((0, 5), np.float32)
            kpss: Optional[np.ndarray] = None
            names: List[Optional[str]] = []
            known_mask: List[bool] = []

            # Detection
            try:
                if args.detection:
                    dets, kpss = scrfd.detect(
                        frame, conf=args.conf, max_num=args.max_num
                    )
            except Exception as e:
                logging.warning(f"[warn] detection failed this frame: {e}")
                dets, kpss = np.zeros((0, 5), np.float32), None

            # Recognition (Top-K with crowd skip)
            names = []
            known_mask = []
            skip_recognition = (
                args.perf_mode
                and ema_ms is not None
                and ema_ms > (target_frame_time * 800)
            )  # 80% of frame time

            if (
                args.recognition
                and not skip_recognition
                and dets is not None
                and dets.shape[0] > 0
                and gal_feats is not None
            ):
                try:
                    Hf, Wf = frame.shape[:2]
                    n_det = dets.shape[0]

                    names_full: List[Optional[str]] = [None] * n_det
                    known_mask = [False] * n_det

                    if n_det > args.crowd_thr:
                        # Too many faces: skip recognition (drawing falls back to score).
                        names = names_full
                    else:
                        sel_idx = (
                            np.arange(n_det, dtype=np.int32)
                            if n_det <= args.recog_topk
                            else pick_topk_indices(
                                dets, topk=args.recog_topk, H=Hf, W=Wf
                            )
                        )

                        crops: List[np.ndarray] = []
                        for i_sel in sel_idx:
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
                                    else cv2.resize(frame, (112, 112))
                                )
                            crops.append(crop)

                        if crops:
                            feats = infer_arc_batched(
                                arc, crops, max_bs=args.recog_topk
                            )  # type: ignore[arg-type]
                            S = feats @ gal_feats.T
                            best_j = np.argmax(S, axis=1)
                            best_v = S[np.arange(S.shape[0]), best_j]
                            for pos, (vv, jj) in enumerate(zip(best_v, best_j)):
                                det_i = int(sel_idx[pos])
                                if vv >= args.sim_thr:
                                    names_full[det_i] = f"{gal_labels[jj]} ({vv:.2f})"
                                    known_mask[det_i] = True
                                else:
                                    names_full[det_i] = "unknown"

                        names = names_full
                except Exception as e:
                    logging.warning(f"[warn] recognition failed this frame: {e}")
                    names = [None] * dets.shape[0]
                    known_mask = [False] * dets.shape[0]
            else:
                # Recognition disabled or no detections: keep arrays aligned with dets
                names = [None] * (0 if dets is None else dets.shape[0])
                known_mask = [False] * (0 if dets is None else dets.shape[0])

            # Ensure alignment even when empty
            if dets is None or dets.shape[0] == 0:
                names = []
                known_mask = []
            else:
                # If recognition didn't run (or was skipped), pad for alignment
                if not names or len(names) != dets.shape[0]:
                    N = int(dets.shape[0])
                    names = [None] * N  # draw_overlays will show score for None
                    known_mask = [False] * N

            # Blur (known/unknown/all)
            if args.blur and dets is not None and dets.shape[0] > 0:
                Hf, Wf = frame.shape[:2]
                for i, (x1, y1, x2, y2, _) in enumerate(dets):
                    apply = True
                    if args.blur_mode != "all" and args.recognition:
                        is_known = known_mask[i] if i < len(known_mask) else False
                        apply = (args.blur_mode == "known" and is_known) or (
                            args.blur_mode == "unknown" and not is_known
                        )
                    if apply:
                        from .pixelate import expand_clip, pixelate_roi

                        x1e, y1e, x2e, y2e = expand_clip(
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                            args.pixel_margin,
                            Wf,
                            Hf,
                        )
                        pixelate_roi(
                            frame,
                            x1e,
                            y1e,
                            x2e,
                            y2e,
                            blocks_on_short=args.pixel_blocks,
                            noise_sigma=args.pixel_noise,
                        )

            # Draw overlays
            if args.draw_boxes or args.draw_names:
                frame = draw_overlays(
                    frame,
                    dets,
                    names,
                    kpss,
                    draw_boxes=args.draw_boxes,
                    draw_names=args.draw_names,
                )

            # Stats / overlay
            dt_ms = (time.perf_counter() - start) * 1000.0
            ema_ms = dt_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * dt_ms)
            if getattr(args, "show_fps", False):
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

            if not args.no_window:
                cv2.imshow("Face Anonymizer", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            current_time = time.perf_counter()
            elapsed = current_time - last_frame_time
            if elapsed < target_frame_time:
                sleep_time = target_frame_time - elapsed
                time.sleep(sleep_time)
            last_frame_time = time.perf_counter()

            if total % max(1, args.print_every) == 0:
                sec = max(1e-9, time.perf_counter() - t0)
                fps_now = total / sec
                logging.info(
                    f"[{total:05d}] frame={dt_ms:.2f} ms  EMA={ema_ms:.2f}  FPS={fps_now:.1f}  faces={0 if dets is None else dets.shape[0]}"
                )

    finally:
        try:
            cap.release()
        except Exception:
            pass

        if rstp_writer is not None:
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
    main()
