# -*- coding: utf-8 -*-
"""
SCRFD TensorRT on Jetson Orin + Pixelation anonymization
- TRT 10.x tensor API (set_tensor_address + execute_async_v3)
- Pinned host buffers, async H2D/D2H, CUDA event timing
- Robust SCRFD postproc (stride 8/16/32)
- Optional NVENC (GStreamer) writer
- Pixelation with adjustable strength/margin/max-faces (+ optional noise)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

try:
    import pycuda.driver as cuda
except ImportError:
    print("PyCUDA not found, make sure to install it for GPU support.")

sys.path.append("/usr/lib/python3/dist-packages")

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not found, make sure to install it.")

# import pycuda.autoinit  # noqa

# ---------------- detection / postproc defaults ----------------
CONF_THRES = 0.50
NMS_IOU = 0.40
IGNORE_SHORT = 8
STRIDES = (8, 16, 32)
NUM_ANCHORS = 2
TOPK_PER_LEVEL = 150
MAX_DETS = 100
# ---------------------------------------------------------------


def letterbox_bgr(img, new=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new[0] / h, new[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (h, w) != (nh, nw):
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    else:
        img_r = img
    canvas = np.full((new[0], new[1], 3), color, dtype=np.uint8)
    top = (new[0] - nh) // 2
    left = (new[1] - nw) // 2
    canvas[top : top + nh, left : left + nw] = img_r
    return canvas, r, (left, top), (w, h)


def nms_numpy(dets, iou=0.5):
    if dets is None or len(dets) == 0:
        return dets
    boxes = dets[:, :4].astype(np.float32)
    scores = dets[:, 4].astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou)[0]
        order = order[inds + 1]
    return dets[keep]


def draw_dets(img, dets, color=(0, 255, 0), thickness=2, put_fps=None, draw_boxes=True):
    if draw_boxes:
        for x1, y1, x2, y2, sc in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    if put_fps:
        cv2.putText(
            img,
            put_fps,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )
    return img


def make_center_priors(size, stride, num_anchors=2):
    fh = size // stride
    fw = size // stride
    xs = (np.arange(fw) + 0.5) * stride
    ys = (np.arange(fh) + 0.5) * stride
    cx, cy = np.meshgrid(xs, ys)
    pts = np.stack([cx.reshape(-1), cy.reshape(-1)], axis=1)
    return np.repeat(pts, repeats=num_anchors, axis=0).astype(np.float32)


def distance2bbox(points, distances):
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def expand_clip(x1, y1, x2, y2, margin, W, H):
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    w, h = (x2 - x1), (y2 - y1)
    w *= 1.0 + margin
    h *= 1.0 + margin
    x1n = max(0, int(cx - w * 0.5))
    y1n = max(0, int(cy - h * 0.5))
    x2n = min(W - 1, int(cx + w * 0.5))
    y2n = min(H - 1, int(cy + h * 0.5))
    return x1n, y1n, x2n, y2n


# ------------------- Pixelation (fast & adjustable) -------------------
def pixelate_roi(img, x1, y1, x2, y2, blocks_on_short=8, noise_sigma=0.0):
    """
    Let [x1:x2, y1:y2] area do the pixelation：
    - blocks_on_short: short block number（smaller value more coarse, stronger privacy）
    - noise_sigma: adding gaussian noise to the pixelation process（0=no noise）
    """
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    short = min(w, h)
    blocks_on_short = max(1, int(blocks_on_short))

    scale = blocks_on_short / float(short)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))

    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if noise_sigma and noise_sigma > 0:
        noise = np.random.normal(0, noise_sigma, size=big.shape).astype(np.float32)
        big = np.clip(big.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img[y1:y2, x1:x2] = big


def apply_pixelation(img, dets, margin=0.25, blocks=8, max_faces=32, noise_sigma=0.0):
    """
    Pixelate multiple face boxes:
    Sort by score and limit the number processed.
    For each box, expand it by margin, then apply pixelation.
    """
    if dets is None or len(dets) == 0:
        return
    H, W = img.shape[:2]
    # 高分在前
    dets_sorted = dets[dets[:, 4].argsort()[::-1]]
    for i, (x1, y1, x2, y2, sc) in enumerate(dets_sorted[:max_faces]):
        x1e, y1e, x2e, y2e = expand_clip(
            int(x1), int(y1), int(x2), int(y2), margin, W, H
        )
        pixelate_roi(
            img, x1e, y1e, x2e, y2e, blocks_on_short=blocks, noise_sigma=noise_sigma
        )


# ------------------- TensorRT wrapper -------------------
class TRTInfer:
    def __init__(self, engine_path, input_name="input.1", size=640, verbose=False):
        self.size = size
        self.verbose = verbose
        # ✅ Create CUDA context in THIS thread (video thread)
        import pycuda.driver as cuda

        cuda.init()
        self.stream = cuda.Stream()  # stream is tied to this thread/context

        logger = trt.Logger(trt.Logger.WARNING)
        with (
            open(os.path.expanduser(engine_path), "rb") as f,
            trt.Runtime(logger) as rt,
        ):
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # tensors
        self.names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.modes = {n: self.engine.get_tensor_mode(n) for n in self.names}
        self.dtypes = {
            n: trt.nptype(self.engine.get_tensor_dtype(n)) for n in self.names
        }

        self.in_name = input_name
        assert self.in_name in self.names, f"input '{self.in_name}' not in {self.names}"
        if not self.ctx.set_input_shape(self.in_name, (1, 3, self.size, self.size)):
            raise RuntimeError("set_input_shape failed")

        self.shapes = {n: tuple(self.ctx.get_tensor_shape(n)) for n in self.names}
        if self.verbose:
            for n in self.names:
                print(
                    f"[tensor] {n:20s} mode={self.modes[n].name} shape={self.shapes[n]} dtype={self.dtypes[n]}"
                )

        # alloc
        self.alloc = {}
        for n in self.names:
            nbytes = int(np.prod(self.shapes[n])) * np.dtype(self.dtypes[n]).itemsize
            if nbytes == 0:
                continue
            mem = cuda.mem_alloc(nbytes)
            self.alloc[n] = mem
            self.ctx.set_tensor_address(n, int(mem))

        self.h_in = cuda.pagelocked_empty(
            (1, 3, self.size, self.size), dtype=self.dtypes[self.in_name]
        )
        self.out_names = [
            n for n in self.names if self.modes[n] == trt.TensorIOMode.OUTPUT
        ]
        self.h_out = {
            n: cuda.pagelocked_empty(int(np.prod(self.shapes[n])), dtype=self.dtypes[n])
            for n in self.out_names
            if int(np.prod(self.shapes[n])) > 0
        }

        self.ev_start = cuda.Event()
        self.ev_end = cuda.Event()

    def _preprocess(self, frame_bgr):
        inp, r, (left, top), (W, H) = letterbox_bgr(frame_bgr, (self.size, self.size))
        x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (x - 127.5) / 128.0
        x = np.transpose(x, (2, 0, 1)).astype(self.h_in.dtype, copy=False)
        self.h_in[...] = x[None]
        return r, left, top, W, H

    def infer(self, frame_bgr):
        r, left, top, W, H = self._preprocess(frame_bgr)
        cuda.memcpy_htod_async(self.alloc[self.in_name], self.h_in, self.stream)

        self.ev_start.record(self.stream)
        if not self.ctx.execute_async_v3(self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.ev_end.record(self.stream)

        outs = {}
        for n in self.out_names:
            if n not in self.alloc or n not in self.h_out:
                continue
            cuda.memcpy_dtoh_async(self.h_out[n], self.alloc[n], self.stream)

        self.stream.synchronize()
        gpu_ms = self.ev_end.time_since(self.ev_start)

        for n in self.out_names:
            if n in self.h_out:
                outs[n] = np.array(self.h_out[n]).reshape(self.shapes[n])

        dets = self._postproc_scrfd(outs, left, top, r, W, H)
        return dets, gpu_ms

    def _postproc_scrfd(self, outs, left, top, r, W, H):
        scores_map = {
            arr.shape[0]: arr.reshape(-1)
            for arr in outs.values()
            if arr.ndim == 2 and arr.shape[-1] == 1
        }
        bboxes_map = {
            arr.shape[0]: arr.reshape(-1, 4)
            for arr in outs.values()
            if arr.ndim == 2 and arr.shape[-1] == 4
        }
        if not scores_map or not bboxes_map:
            return np.zeros((0, 5), dtype=np.float32)

        all_scores, all_boxes = [], []
        for stride in STRIDES:
            fh = self.size // stride
            fw = self.size // stride
            length = fh * fw * NUM_ANCHORS
            if length not in scores_map or length not in bboxes_map:
                continue
            sc = scores_map[length]
            bb = bboxes_map[length]

            # logits or probs 自适应
            if not (0.0 <= sc.min() and sc.max() <= 1.0):
                sc = 1.0 / (1.0 + np.exp(-sc))

            A = NUM_ANCHORS
            sc2 = sc.reshape(fh * fw, A)
            bb2 = bb.reshape(fh * fw, A, 4)
            idx = np.argmax(sc2, axis=1)
            sc1 = sc2[np.arange(fh * fw), idx]
            bb1 = bb2[np.arange(fh * fw), idx, :]

            if np.median(bb1) < 12.0:
                bb1 = bb1 * float(stride)

            pri = make_center_priors(self.size, stride, 1)
            boxes_xyxy = distance2bbox(pri, bb1.astype(np.float32))

            keep = sc1 >= getattr(self, "conf_thres", CONF_THRES)
            if not np.any(keep):
                continue
            sc1 = sc1[keep]
            boxes_xyxy = boxes_xyxy[keep]

            topk = int(getattr(self, "topk_per_level", TOPK_PER_LEVEL))
            if sc1.size > topk:
                idk = np.argpartition(sc1, -topk)[-topk:]
                sc1 = sc1[idk]
                boxes_xyxy = boxes_xyxy[idk]

            all_scores.append(sc1)
            all_boxes.append(boxes_xyxy)

        if not all_scores:
            return np.zeros((0, 5), dtype=np.float32)

        scores = np.concatenate(all_scores, axis=0)
        boxes = np.concatenate(all_boxes, axis=0)

        gtopk = max(
            int(getattr(self, "topk_per_level", TOPK_PER_LEVEL)) * len(STRIDES),
            int(getattr(self, "max_dets", MAX_DETS)) * 4,
        )
        if scores.size > gtopk:
            idg = np.argpartition(scores, -gtopk)[-gtopk:]
            scores = scores[idg]
            boxes = boxes[idg]

        dets = []
        for b, s in zip(boxes, scores):
            x1 = (b[0] - left) / r
            y1 = (b[1] - top) / r
            x2 = (b[2] - left) / r
            y2 = (b[3] - top) / r
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W - 1, x2)))
            y2 = int(max(0, min(H - 1, y2)))
            if min(x2 - x1, y2 - y1) < IGNORE_SHORT:
                continue
            dets.append([x1, y1, x2, y2, float(s)])

        if not dets:
            return np.zeros((0, 5), dtype=np.float32)
        dets = np.array(dets, dtype=np.float32)
        dets = nms_numpy(dets, iou=NMS_IOU)

        K = int(getattr(self, "max_dets", MAX_DETS))
        if dets.shape[0] > K:
            dets = dets[dets[:, 4].argsort()[::-1][:K]]
        return dets


# -------------------- NVENC writer helpers --------------------
def open_nvenc_writer(out_path: str, width: int, height: int, fps_in: float):
    """
    NVENC writer via GStreamer for OpenCV.
    - DO NOT use '-e' here (only for gst-launch).
    - Keep caps on appsrc; convert to NV12 on NVMM before nvv4l2h264enc.
    """
    fps = float(fps_in) if fps_in and fps_in > 0 else 25.0
    fps_i = int(round(fps))

    candidates = [
        # Preferred: sysmem(BGR) -> videoconvert -> nvvidconv -> NVMM/NV12 -> NVENC -> mp4mux
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! queue leaky=downstream max-size-buffers=1 "
            f"! videoconvert "
            f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 "
            f"! h264parse ! mp4mux "
            f"! filesink location={out_path} sync=false"
        ),
        # Minimal variant (some OpenCV builds accept this fine)
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! videoconvert "
            f"! nvv4l2h264enc insert-sps-pps=true "
            f"! h264parse ! mp4mux "
            f"! filesink location={out_path} sync=false"
        ),
        # Fallback to MKV container (very tolerant)
        (
            f"appsrc caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps_i}/1 "
            f"! videoconvert "
            f"! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 "
            f"! nvv4l2h264enc insert-sps-pps=true "
            f"! h264parse ! matroskamux "
            f"! filesink location={os.path.splitext(out_path)[0]}.mkv sync=false"
        ),
    ]

    for p in candidates:
        w = cv2.VideoWriter(p, cv2.CAP_GSTREAMER, 0, fps, (width, height))
        if w.isOpened():
            print("[nvenc] using pipeline:", p)
            return w
    return None


# --------------------------- main ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # detection knobs

    """
    Missed detections: lower conf_thres (≈0.35–0.45) or increase topk_per_level.

    False positives: raise conf_thres, lower topk_per_level, or lower NMS_IOU (e.g., 0.35).

    Duplicate/overlapping boxes: lower NMS_IOU further; also reduce per-level/global top-k.

    Many tiny specks: increase IGNORE_SHORT (≈10–12).

    Performance pressure (fps): reduce topk_per_level and max_dets; optionally increase print_every to cut log overhead.
    """
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--max_dets", type=int, default=None)

    # io / engine
    ap.add_argument("--engine", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--input_name", default="input.1")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--nvenc", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # pixelation knobs
    ap.add_argument(
        "--pixelate", action="store_true", help="enable pixelation anonymization"
    )
    ap.add_argument(
        "--pixel_blocks",
        type=int,
        default=8,
        help="short-side blocks (smaller = stronger)",
    )
    ap.add_argument(
        "--pixel_margin",
        type=float,
        default=0.25,
        help="bbox expand ratio before pixelation",
    )
    ap.add_argument(
        "--pixel_max_faces", type=int, default=32, help="max faces pixelated per frame"
    )
    ap.add_argument(
        "--pixel_noise",
        type=float,
        default=0.0,
        help="add noise after pixelation (0=off)",
    )
    ap.add_argument("--no_boxes", action="store_true", help="do not draw rectangles")

    args = ap.parse_args()

    eng = os.path.expanduser(args.engine)
    assert os.path.exists(eng), f"engine not found: {eng}"

    # Input: auto choose CAP_GSTREAMER if looks like a pipeline
    inp = args.input.strip()
    # use_gst = ('!' in inp) and inp.endswith('appsink drop=true max-buffers=1 sync=false')
    # For debug purpose, remove sync=false (not following playback, run the video as fast as it can),
    # keep appsink drop=true max-buffers=1: if consumer not timely pick up the frame, it will be drop and only keep at most one frame
    use_gst = ("!" in inp) and inp.endswith("appsink drop=true max-buffers=1")
    cap = cv2.VideoCapture(inp, cv2.CAP_GSTREAMER) if use_gst else cv2.VideoCapture(inp)
    assert cap.isOpened(), f"cannot open input: {args.input}"
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Output writer
    writer = None
    if args.out:
        out_path = os.path.expanduser(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        if args.nvenc:
            w = open_nvenc_writer(out_path, W, H, fps_in)
            if w.isOpened():
                writer = w
            else:
                print("[warn] NVENC pipeline failed to open, falling back to CPU mp4v")
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                out_path, fourcc, fps_in if fps_in > 0 else 25.0, (W, H)
            )
            if not writer.isOpened():
                raise RuntimeError("CPU VideoWriter failed to open")

    infer = TRTInfer(
        engine_path=eng,
        input_name=args.input_name,
        size=args.size,
        verbose=args.verbose,
    )
    if args.conf is not None:
        infer.conf_thres = args.conf
    if args.topk is not None:
        infer.topk_per_level = args.topk
    if args.max_dets is not None:
        infer.max_dets = args.max_dets

    # warmup
    for _ in range(8):
        ok, f = cap.read()
        if not ok:
            break
        _ = infer.infer(f)

    total_frames = 0
    t0 = time.perf_counter()
    ema_ms = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # dets (x1,y1,x2,y2,score)
        dets, gpu_ms = infer.infer(frame)
        total_frames += 1
        ema_ms = gpu_ms if ema_ms is None else (0.9 * ema_ms + 0.1 * gpu_ms)

        # --- pixelation ---
        if args.pixelate and dets is not None and len(dets) > 0:
            apply_pixelation(
                frame,
                dets,
                margin=args.pixel_margin,
                blocks=args.pixel_blocks,
                max_faces=args.pixel_max_faces,
                noise_sigma=args.pixel_noise,
            )

        # overlay & write
        if writer is not None:
            elapsed = time.perf_counter() - t0
            fps_now = total_frames / elapsed if elapsed > 0 else 0.0
            overlay = f"GPU {gpu_ms:.2f} ms | EMA {ema_ms:.2f} ms | FPS {fps_now:.1f} | faces {len(dets)}"
            out_frame = draw_dets(
                frame, dets, put_fps=overlay, draw_boxes=(not args.no_boxes)
            )
            writer.write(out_frame)

        if total_frames % args.print_every == 0:
            elapsed = time.perf_counter() - t0
            fps_now = total_frames / elapsed if elapsed > 0 else 0.0
            print(
                f"[{total_frames:05d}] GPU={gpu_ms:.2f} ms  EMA={ema_ms:.2f} ms  FPS={fps_now:.1f}  faces={len(dets)}"
            )

    cap.release()
    if writer is not None:
        writer.release()

    if total_frames:
        total_time = time.perf_counter() - t0
        print(
            f"done. frames={total_frames} avg_fps={total_frames / total_time:.2f} avg_gpu_ms≈{ema_ms:.2f}"
        )


if __name__ == "__main__":
    main()


""" USAGE
1. Using Orin builtin GPU supported GStreamer to decode video faster!
python scrfd_trt_pixelate.py
    --engine "$HOME/anon-orin/models/scrfd_2.5g_640.engine"   \
    --input  "filesrc location=/home/openmind/Desktop/wenjinf-OM-workspace/videos/my_video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder enable-max-performance=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    --out    "$HOME/anon-orin/results/out_nvenc_mask.mp4"
    --nvenc
    -conf 0.5
    --topk 100
    --max_dets 50
    --pixelate
    --pixel_blocks 8
    --pixel_margin 0.25

2. Using default ffmpeg CV2 to decode video on CPU much slower!
python scrfd_trt_pixelate.py
    --engine "$HOME/anon-orin/models/scrfd_2.5g_640.engine"   \
    --input  "/home/openmind/Desktop/wenjinf-OM-workspace/videos/my_video.mp4"
    --out    "$HOME/anon-orin/results/out_nvenc_mask.mp4"
    -conf 0.5
    --topk 100
    --max_dets 50
    --pixelate
    --pixel_blocks 8
    --pixel_margin 0.25
"""

""" Example outputs AVG FPS: 32
Opening in BLOCKING MODE
NvMMLiteOpen : Block : BlockType = 261
NvMMLiteBlockCreate : Block : BlockType = 261
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (1063) open OpenCV | GStreamer warning: unable to query duration of stream
[ WARN:0] global ./modules/videoio/src/cap_gstreamer.cpp (1100) open OpenCV | GStreamer warning: Cannot query video position: status=1, value=1, duration=-1
Opening in BLOCKING MODE
[nvenc] using pipeline: appsrc caps=video/x-raw,format=BGR,width=1920,height=1080,framerate=10/1 ! queue leaky=downstream max-size-buffers=1 ! videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! nvv4l2h264enc insert-sps-pps=true maxperf-enable=1 control-rate=1 bitrate=4000000 ! h264parse ! mp4mux ! filesink location=/home/openmind/anon-orin/results/out_nvenc_mask.mp4 sync=false
NvMMLiteOpen : Block : BlockType = 4
===== NvVideo: NVENC =====
NvMMLiteBlockCreate : Block : BlockType = 4
[08/25/2025-15:15:56] [TRT] [W] Using an engine plan file across different models of devices is not recommended and is likely to affect performance or even cause errors.
H264: Profile = 66 Level = 0
NVMEDIA: Need to set EMC bandwidth : 282000
[00010] GPU=7.81 ms  EMA=8.26 ms  FPS=33.6  faces=1
[00020] GPU=7.74 ms  EMA=8.31 ms  FPS=33.9  faces=0
done. frames=28 avg_fps=32.07 avg_gpu_ms≈8.57
"""
