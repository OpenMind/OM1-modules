"""
Standalone test script for YOLO11 pose detection + fall detection.

Run this to verify pose/fall detection works before integrating into the main pipeline.

Usage:
  python test_pose.py --engine /path/to/yolo11s-pose.engine --device /dev/video0
  python test_pose.py --engine /path/to/yolo11s-pose.engine --device /dev/video0 --rtsp rtsp://localhost:8554/pose_test

Keys:
  ESC or 'q' - quit
  's' - toggle skeleton drawing
  'b' - toggle bounding boxes
  'f' - toggle fall detection overlay
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from .yolo_pose import TRTYOLOPose
from .fall_detector import FallDetector, FallStatus
from .draw_pose import draw_pose_overlays, draw_fall_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser("Test YOLO11 Pose + Fall Detection")
    ap.add_argument("--engine", required=True, help="Path to yolo11s-pose.engine")
    ap.add_argument("--device", default="/dev/video0", help="Camera device or video file")
    ap.add_argument("--width", type=int, default=1280, help="Capture width")
    ap.add_argument("--height", type=int, default=720, help="Capture height")
    ap.add_argument("--fps", type=int, default=30, help="Capture FPS")
    ap.add_argument("--size", type=int, default=640, help="Model input size")
    ap.add_argument("--conf", type=float, default=0.5, help="Detection confidence")
    ap.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    ap.add_argument("--max-num", type=int, default=10, help="Max detections")
    ap.add_argument("--rtsp", default=None, help="Optional RTSP URL to stream to")
    ap.add_argument("--no-window", action="store_true", help="Disable display window")
    args = ap.parse_args()

    # Load pose detector
    log.info("Loading pose engine: %s", args.engine)
    pose = TRTYOLOPose(
        args.engine,
        size=args.size,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
    )
    log.info("Pose engine loaded successfully")

    # Fall detector
    fall_det = FallDetector(
        horizontal_ratio_thr=0.4,
        height_ratio_thr=0.3,
        aspect_ratio_thr=1.2,
        temporal_frames=5,
    )
    log.info("Fall detector initialized")

    # Open camera
    log.info("Opening camera: %s", args.device)
    if args.device.isdigit():
        cap = cv2.VideoCapture(int(args.device))
    else:
        cap = cv2.VideoCapture(args.device)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    if not cap.isOpened():
        log.error("Failed to open camera: %s", args.device)
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("Camera opened: %dx%d @ %.1f FPS", actual_w, actual_h, actual_fps)

    # Optional RTSP writer
    rtsp_writer = None
    if args.rtsp:
        try:
            from rtsp_video_writer import RTSPVideoStreamWriter
            rtsp_writer = RTSPVideoStreamWriter(
                actual_w, actual_h, int(actual_fps), args.rtsp, None
            )
            log.info("RTSP streaming to: %s", args.rtsp)
        except Exception as e:
            log.warning("Could not setup RTSP: %s", e)

    # Drawing toggles
    draw_skeleton = True
    draw_boxes = True
    draw_fall = True

    # Stats
    frame_count = 0
    t0 = time.perf_counter()
    ema_ms: Optional[float] = None

    log.info("Starting inference loop. Press ESC or 'q' to quit.")
    log.info("Keys: 's'=skeleton, 'b'=boxes, 'f'=fall overlay")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                log.warning("Frame read failed, retrying...")
                time.sleep(0.01)
                continue

            frame_count += 1
            t_start = time.perf_counter()

            # Run pose detection
            try:
                dets, kps = pose.detect(frame, conf=args.conf, max_num=args.max_num)
            except Exception as e:
                log.error("Pose detection error: %s", e)
                dets, kps = np.zeros((0, 5), np.float32), np.zeros((0, 17, 3), np.float32)

            # Run fall detection
            fall_statuses: List[FallStatus] = []
            if dets is not None and len(dets) > 0:
                fall_statuses = fall_det.detect_batch(dets, kps)

            # Calculate timing
            t_infer = time.perf_counter()
            infer_ms = (t_infer - t_start) * 1000.0

            # Draw overlays
            frame = draw_pose_overlays(
                frame,
                dets,
                kps,
                fall_statuses=fall_statuses if draw_fall else None,
                draw_skeleton=draw_skeleton,
                draw_boxes=draw_boxes,
                draw_fall_status=draw_fall,
                kp_conf_thr=0.5,
            )

            # Draw fall alert banner
            if draw_fall and fall_statuses:
                fall_count = sum(1 for s in fall_statuses if s.is_fallen)
                if fall_count > 0:
                    draw_fall_alert(frame, fall_count)

            # Calculate FPS
            t_draw = time.perf_counter()
            dt_ms = (t_draw - t_start) * 1000.0
            ema_ms = dt_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * dt_ms)
            elapsed = max(1e-9, t_draw - t0)
            fps_now = frame_count / elapsed

            # Count fallen
            n_poses = 0 if dets is None else len(dets)
            n_fallen = sum(1 for s in fall_statuses if s.is_fallen) if fall_statuses else 0

            # Build overlay text
            overlay = (
                f"{dt_ms:.1f} ms (EMA {ema_ms:.1f}) | "
                f"{fps_now:.1f} FPS | "
                f"poses {n_poses} | "
                f"fallen {n_fallen}"
            )

            # Draw FPS overlay
            cv2.putText(
                frame, overlay, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 220, 40), 2, cv2.LINE_AA
            )

            # Draw toggle status
            toggle_text = f"[s]keleton:{draw_skeleton} [b]ox:{draw_boxes} [f]all:{draw_fall}"
            cv2.putText(
                frame, toggle_text, (10, actual_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

            # Draw individual fall status info
            if fall_statuses:
                y_offset = 60
                for i, status in enumerate(fall_statuses):
                    color = (0, 0, 255) if status.is_fallen else (0, 255, 0)
                    state = "FALLEN" if status.is_fallen else "OK"
                    info = f"Person {i}: {state} ({status.confidence:.2f}) - {status.reason[:40]}"
                    cv2.putText(
                        frame, info, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                    )
                    y_offset += 20

            # Stream to RTSP
            if rtsp_writer:
                rtsp_writer.write_frame(frame)

            # Display
            if not args.no_window:
                cv2.imshow("Pose + Fall Detection Test", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord('q'):
                    log.info("Quit requested")
                    break
                elif key == ord('s'):
                    draw_skeleton = not draw_skeleton
                    log.info("Skeleton: %s", draw_skeleton)
                elif key == ord('b'):
                    draw_boxes = not draw_boxes
                    log.info("Boxes: %s", draw_boxes)
                elif key == ord('f'):
                    draw_fall = not draw_fall
                    log.info("Fall overlay: %s", draw_fall)

            # Log periodically
            if frame_count % 30 == 0:
                log.info(
                    "[%05d] infer=%.1fms total=%.1fms EMA=%.1fms FPS=%.1f poses=%d fallen=%d",
                    frame_count, infer_ms, dt_ms, ema_ms, fps_now, n_poses, n_fallen
                )

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        cap.release()
        if rtsp_writer:
            rtsp_writer.stop()
        if not args.no_window:
            cv2.destroyAllWindows()

    log.info("Done. Processed %d frames in %.1f seconds (avg %.1f FPS)",
             frame_count, time.perf_counter() - t0,
             frame_count / max(1e-9, time.perf_counter() - t0))


if __name__ == "__main__":
    main()