#!/usr/bin/env python3
"""
Usage
python -m face_recog_stream.realtime_stream --scrfd-engine "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/scrfd_2.5g_bnkps_shape640x640.engine" --arc-engine "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/buffalo_m_w600k_r50.engine" --gallery "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/gallery" --gst --device /dev/video0 --width 1280 --height 720 --fps 30 --detection --recognition --blur --blur-mode all --draw-boxes --draw-names --show-fps --recog-topk 8 --crowd-thr 12 --rtmp "rtmp://localhost:1935/live" --no-window --nvenc
 python -m face_recog_stream.realtime_stream   --scrfd-engine "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/scrfd_2.5g_bnkps_shape640x640.engine"   --arc-engine   "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/models/buffalo_m_w600k_r50.engine"   --gallery      "/home/openmind/Desktop/wenjinf-OM-workspace/OM1-modules/src/om1_vlm/anonymizationSys/gallery"   --gst --device /dev/video0   --width 1280 --height 720 --fps 30   --detection --recognition --blur --blur-mode all   --draw-boxes --draw-names --show-fps   --recog-topk 8 --crowd-thr 12   --rtmp "rtmp://api-video-ingest.openmind.org:1935/78e8f78e86ea6b3d?api_key=om1_live_78e8f78e86ea6b3dd5a519dd5a9d91f16711b68c6a3b037748a840807f4b80e4ea05c9709b58ea73"
"""

from __future__ import annotations

import argparse
import signal
import time
from typing import List, Optional

import cv2
import numpy as np

# TensorRT & CUDA contexts
import pycuda.autoinit  # noqa: F401
from face_recog_stream.arcface import TRTArcFace, warp_face_by_5p
from face_recog_stream.draw import draw_overlays
from face_recog_stream.gallery import build_gallery_embeddings
from face_recog_stream.io import (
    build_cam_capture,
    build_file_writer,
    build_gst_capture,
    open_nvenc_rtmp_writer,
)
from face_recog_stream.scrfd import TRTSCRFD
from face_recog_stream.utils import infer_arc_batched, pick_topk_indices


def main() -> None:
    """Run real‑time detection/recognition/blur and optional RTMP/file output.

    CLI arguments configure engines, thresholds, camera, and outputs.
    """
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
        help="If the number of detected faces in a frame exceeds this threshold, skip recognition and only blur/draw boxes.",
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
        "--rtmp", default="", help="RTMP URL to publish (empty = disabled)."
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

    scrfd = TRTSCRFD(args.scrfd_engine, input_name=args.scrfd_input, size=args.size)
    scrfd.nms_thresh = args.nms

    arc = None
    gal_feats = None
    gal_labels: List[str] = []
    if args.recognition:
        arc = TRTArcFace(args.arc_engine, input_name=args.arc_input)
        print(">> Building gallery embeddings…")
        t0 = time.time()
        gal_feats, gal_labels = build_gallery_embeddings(
            args.gallery, scrfd, arc, det_conf=args.conf
        )
        print(f"   identities={len(gal_labels)}  time={time.time() - t0:.2f}s")

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

    rtmp_writer = None
    if args.rtmp:
        rtmp_writer = open_nvenc_rtmp_writer(args.rtmp, W, H, fps_in)
        if rtmp_writer is None:
            raise RuntimeError(
                "RTMP pipeline failed to open (NVENC & x264 both failed)"
            )

    file_writer = (
        build_file_writer(args.outfile, W, H, fps_in, args.nvenc)
        if args.outfile
        else None
    )

    running = True

    def handle_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_sigint)

    t0 = time.perf_counter()
    total = 0
    ema_ms = None

    while running:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        start = time.perf_counter()

        dets = np.zeros((0, 5), np.float32)
        kpss = None
        names: List[str] = []
        known_mask = []

        if args.detection:
            dets, kpss = scrfd.detect(frame, conf=args.conf, max_num=args.max_num)

        # ---------- Recognition (new code) ----------
        # Notes:
        # - If number of faces > args.crowd_thr: skip recognition; mark as "unknown"
        #   (or let the drawing logic fall back to showing the score).
        # - If there are 1..args.crowd_thr faces: run recognition only on up to
        #   args.recog_topk "most important" faces.
        #   Importance = (box area × confidence) with preference for faces closer to
        #   the image center.
        # --- RECOGNITION (Top-K / crowd mode, batch <= args.recog_topk) ---

        names: List[Optional[str]] = []
        known_mask = []
        if args.recognition and dets is not None and dets.shape[0] > 0:
            Hf, Wf = frame.shape[:2]
            n_det = dets.shape[0]

            # 默认：全部未知（None 让 draw_overlays 回落显示 score）
            names_full: List[Optional[str]] = [None] * n_det
            known_mask = [False] * n_det

            if n_det > args.crowd_thr:
                names = names_full
            else:
                if n_det <= args.recog_topk:
                    sel_idx = np.arange(n_det, dtype=np.int32)
                else:
                    sel_idx = pick_topk_indices(dets, topk=args.recog_topk, H=Hf, W=Wf)

                crops: List[np.ndarray] = []
                for i_sel in sel_idx:
                    if kpss is not None:
                        crop = warp_face_by_5p(frame, kpss[i_sel], 112)
                    else:
                        x1, y1, x2, y2, _ = dets[i_sel].astype(int)
                        face = frame[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                        crop = (
                            cv2.resize(face, (112, 112))
                            if face.size > 0
                            else cv2.resize(frame, (112, 112))
                        )
                    crops.append(crop)

                if crops:
                    feats = infer_arc_batched(arc, crops, max_bs=args.recog_topk)
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

        # ---------- Recognition ----------

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
                    from face_recog_stream.pixelate import expand_clip, pixelate_roi

                    x1e, y1e, x2e, y2e = expand_clip(
                        int(x1), int(y1), int(x2), int(y2), args.pixel_margin, Wf, Hf
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

        if args.draw_boxes or args.draw_names:
            frame = draw_overlays(
                frame,
                dets,
                names,
                kpss,
                draw_boxes=args.draw_boxes,
                draw_names=args.draw_names,
            )

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

        if rtmp_writer is not None:
            rtmp_writer.write(frame)
        if file_writer is not None:
            file_writer.write(frame)
        if not args.no_window:
            cv2.imshow("Face Anonymizer", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        if total % max(1, args.print_every) == 0:
            sec = max(1e-9, time.perf_counter() - t0)
            fps_now = total / sec
            print(
                f"[{total:05d}] frame={dt_ms:.2f} ms  EMA={ema_ms:.2f}  FPS={fps_now:.1f}  faces={0 if dets is None else dets.shape[0]}"
            )

    cap.release()
    if rtmp_writer is not None:
        rtmp_writer.release()
    if file_writer is not None:
        file_writer.release()
    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
