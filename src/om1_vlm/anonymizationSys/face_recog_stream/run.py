#!/usr/bin/env python3
"""
Usage examples
--------------
Local mediamtx (RTMP), no preview window, NVENC:
  python -m om1_vlm.anonymizationSys.face_recog_stream.run  \
    --scrfd-engine "/path/to/scrfd_2.5g_bnkps_shape640x640.engine" \
    --arc-engine   "/path/to/buffalo_m_w600k_r50.engine" \
    --gallery      "/path/to/gallery" \
    --gst --device /dev/video0 \
    --width 1280 --height 720 --fps 30 \
    --detection --recognition --blur --blur-mode all \
    --draw-boxes --draw-names --show-fps \
    --recog-topk 8 --crowd-thr 12 \
    --rtmp "rtmp://localhost:1935/live/om1" \
    --no-window --nvenc

OpenMind ingest (headless), NVENC:
  python -m om1_vlm.anonymizationSys.face_recog_stream.run   --scrfd-engine "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/scrfd_2.5g_bnkps_shape640x640.engine"   --arc-engine   "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/buffalo_m_w600k_r50.engine"   --gallery      "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/gallery"   --gst --device /dev/video0   --width 1280 --height 720 --fps 30   --detection --recognition --blur --blur-mode all   --draw-boxes --draw-names --show-fps   --recog-topk 8 --crowd-thr 12   --rtmp "rtmp://api-video-ingest.openmind.org:1935/<OM_API_KEY_ID>?api_key=<OM_API_KEY>"

python -m om1_vlm.anonymizationSys.face_recog_stream.run    --scrfd-engine "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/scrfd_2.5g_bnkps_shape640x640.engine"   --arc-engine   "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/buffalo_m_w600k_r50.engine"   --gallery      "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/gallery"   --gst --device /dev/video0   --width 1280 --height 720 --fps 30   --detection --recognition --blur --blur-mode all   --draw-boxes --draw-names --show-fps   --recog-topk 8 --crowd-thr 12   --rtmp "rtmp://api-video-ingest.openmind.org:1935/<OM_API_KEY_ID>?api_key=<OM_API_KEY>   --no-window --nvenc"
"""

from __future__ import annotations

import argparse
import signal
import time
from typing import List, Optional
import logging

import cv2
import numpy as np

# Ensure CUDA context exists for TensorRT
import pycuda.autoinit  # noqa: F401

from .arcface import TRTArcFace, warp_face_by_5p
from .draw import draw_overlays
from .gallery import build_gallery_embeddings
from .io import (
    build_cam_capture,
    build_file_writer,
    build_gst_capture,
    open_nvenc_rtmp_writer,
    safe_read,
    reopen_capture,
    AsyncVideoWriter,
)
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
    # setup_logging("face_recog_stream", logging_config=get_logging_config())
    logging.info("Starting realtime_stream...")

    ap = argparse.ArgumentParser(
        "Jetson real-time detection + recognition + blur + streaming"
    )
    # Engines
    ap.add_argument(
        "--scrfd-engine", required=True, help="Path to SCRFD TensorRT engine (.engine)."
    )
    ap.add_argument(
        "--scrfd-input", default=None, help="SCRFD input tensor name (optional)."
    )
    ap.add_argument(
        "--size", type=int, default=640, help="SCRFD input size (square; e.g., 640)."
    )
    ap.add_argument(
        "--arc-engine",
        required=False,
        help="Path to ArcFace TensorRT engine (.engine).",
    )
    ap.add_argument(
        "--arc-input", default=None, help="ArcFace input tensor name (optional)."
    )
    ap.add_argument(
        "--gallery",
        default=None,
        help="Gallery root (subfolders per identity). Required if --recognition.",
    )
    # Features
    ap.add_argument("--detection", action="store_true", help="Enable face detection.")
    ap.add_argument(
        "--recognition",
        action="store_true",
        help="Enable face recognition (uses gallery).",
    )
    ap.add_argument(
        "--blur", action="store_true", help="Enable pixelation blur on faces."
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
    # Input (camera/GStreamer)
    ap.add_argument(
        "--cam-index", type=int, default=0, help="V4L2 camera index (0,1,...)."
    )
    ap.add_argument(
        "--device",
        default="/dev/video0",
        help="GStreamer v4l2src device path (used with --gst).",
    )
    ap.add_argument(
        "--gst",
        action="store_true",
        help="Use GStreamer v4l2src pipeline instead of OpenCV V4L2.",
    )
    ap.add_argument("--width", type=int, default=1280, help="Capture width.")
    ap.add_argument("--height", type=int, default=720, help="Capture height.")
    ap.add_argument("--fps", type=int, default=30, help="Capture frame rate.")
    # Outputs
    ap.add_argument(
        "--rtmp", default="", help="RTMP/RTMPS URL to publish (empty = disabled)."
    )
    ap.add_argument(
        "--nvenc", action="store_true", help="Use Jetson NVENC in GStreamer writers."
    )
    ap.add_argument("--outfile", default="", help="Optional MP4 output path.")
    ap.add_argument(
        "--no-window", action="store_true", help="Disable display window (headless)."
    )
    ap.add_argument("--print-every", type=int, default=30, help="Log every N frames.")

    args = ap.parse_args()

    if args.recognition and (not args.arc_engine or not args.gallery):
        raise SystemExit("--recognition requires --arc-engine and --gallery")

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
    cap = (
        build_gst_capture(args.device, args.width, args.height, args.fps)
        if args.gst
        else build_cam_capture(args.cam_index, args.width, args.height, args.fps)
    )
    if not cap or not cap.isOpened():
        raise SystemExit(
            "Failed to open camera. Try --gst or correct device/cam-index."
        )

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    fps_in = cap.get(cv2.CAP_PROP_FPS) or float(args.fps)

    # Open outputs
    rtmp_writer: Optional[AsyncVideoWriter] = None
    if args.rtmp:
        raw_writer = open_nvenc_rtmp_writer(args.rtmp, W, H, fps_in)
        if raw_writer is None:
            logging.warning(
                "[warn] RTMP pipeline failed to open. Continue without streaming."
            )
        else:
            rtmp_writer = AsyncVideoWriter(raw_writer, queue_size=1)

    file_writer = (
        build_file_writer(args.outfile, W, H, fps_in, args.nvenc)
        if args.outfile
        else None
    )
    if args.outfile and file_writer is None:
        logging.warning(
            f"[warn] File writer failed to open: {args.outfile}. Continue without local recording."
        )

    # Graceful shutdown on Ctrl-C
    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        logging.info("SIGINT received, shutting down....")
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.perf_counter()
    total = 0
    ema_ms: Optional[float] = None

    try:
        while running:
            # Robust read with small retries; if still failing, try to reopen once.
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = safe_read(cap, max_retries=5, sleep_sec=0.02)
            if not ok or frame is None:
                logging.warning("[warn] capture read failed, trying to reopen...")
                cap = reopen_capture(
                    cap,
                    build_gst_capture if args.gst else build_cam_capture,
                    args.device if args.gst else args.cam_index,
                    args.width,
                    args.height,
                    args.fps,
                )
                if not cap or not cap.isOpened():
                    logging.error("[error] reopen capture failed, exiting loop.")
                    break
                ok, frame = safe_read(cap, max_retries=5, sleep_sec=0.02)
                if not ok or frame is None:
                    logging.error("[error] still cannot read after reopen, exiting.")
                    break

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
            if (
                args.recognition
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
                            feats = infer_arc_batched(arc, crops, max_bs=args.recog_topk)  # type: ignore[arg-type]
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

            # Outputs (non-blocking RTMP)
            if rtmp_writer is not None and rtmp_writer.is_open():
                rtmp_writer.write(frame)
            if file_writer is not None:
                file_writer.write(frame)

            # Optional preview
            if not args.no_window:
                cv2.imshow("Face Anonymizer", frame)
                # Allow ESC to exit even when not receiving signals
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if total % max(1, args.print_every) == 0:
                sec = max(1e-9, time.perf_counter() - t0)
                fps_now = total / sec
                logging.info(
                    f"[{total:05d}] frame={dt_ms:.2f} ms  EMA={ema_ms:.2f}  FPS={fps_now:.1f}  faces={0 if dets is None else dets.shape[0]}"
                )

    finally:
        # Cleanup
        try:
            cap.release()
        except Exception:
            pass
        if rtmp_writer is not None:
            rtmp_writer.close()
        if file_writer is not None:
            try:
                file_writer.release()
            except Exception:
                pass
        if not args.no_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
