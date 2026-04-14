"""
Model Manager for Face Recognition Stream System.

Handles automatic TensorRT engine compilation and version checking.
If an engine file is missing or incompatible with the current TensorRT
runtime, it recompiles from the corresponding ONNX file automatically.

Models managed:
  - SCRFD 10G       : face detection      (fixed 1×3×640×640)
  - AdaFace IR-101  : face recognition    (dynamic batch 1-16×3×112×112)
  - YOLO11s-pose    : pose estimation     (fixed 1×3×640×640)

Usage
-----
    from .model_manager import ModelManager

    mm = ModelManager(
        model_dir="/path/to/onnx/models",
        engine_dir="/path/to/engines",
    )
    scrfd_engine  = mm.ensure_engine("scrfd_10g")
    arc_engine    = mm.ensure_engine("adaface_ir101")
    pose_engine   = mm.ensure_engine("yolo11s_pose")

CLI
---
    python -m om1_vlm.anonymizationSys.face_recog_stream.model_manager --list
    python -m om1_vlm.anonymizationSys.face_recog_stream.model_manager --prepare all
    python -m om1_vlm.anonymizationSys.face_recog_stream.model_manager --prepare scrfd_10g
    python -m om1_vlm.anonymizationSys.face_recog_stream.model_manager --force-recompile --prepare all
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import tensorrt as trt

logger = logging.getLogger(__name__)

# Model Registry
# Each entry describes how to compile one ONNX → engine.
#
#   onnx_name    : filename looked up in model_dir
#   engine_name  : filename written to engine_dir
#   fp16         : compile with --fp16
#   input_name   : ONNX input tensor name (check with `onnxruntime` or Netron)
#   dynamic      : if present, use min/opt/max shapes for trtexec
#   fixed_shape  : if present (and no dynamic), use fixed input shape

MODEL_REGISTRY: Dict[str, dict] = {
    "scrfd_10g": {
        "onnx_name": "det_10g.onnx",
        "engine_name": "scrfd_10g.engine",
        "description": "SCRFD 10G face detector",
        "fp16": True,
        "workspace": 4096,
        "input_name": "input.1",
        "fixed_shape": "1x3x640x640",
    },
    "adaface_ir101": {
        "onnx_name": "adaface_ir101.onnx",
        "engine_name": "adaface_ir101.engine",
        "description": "AdaFace IR-101 face recognition (512-d)",
        "fp16": True,
        "workspace": 4096,
        "input_name": "input",
        "dynamic": {
            "min": "1x3x112x112",
            "opt": "4x3x112x112",
            "max": "16x3x112x112",
        },
    },
    "yolo11s_pose": {
        "onnx_name": "yolo11s-pose.onnx",
        "engine_name": "yolo11s-pose.engine",
        "description": "YOLO11s pose estimation (17 keypoints)",
        "fp16": True,
        "workspace": 4096,
        # No shape args — this ONNX has static shapes baked in
    },
}


# ModelManager
class ModelManager:
    """
    Manages TensorRT engine lifecycle for the face-recognition pipeline.

    - Checks if an engine file exists and is loadable by the current TRT runtime.
    - If missing or incompatible, recompiles from the ONNX source automatically.
    """

    def __init__(
        self,
        model_dir: str,
        engine_dir: str,
        trtexec_path: Optional[str] = None,
        force_recompile: bool = False,
        platform_prefix: str = "",
    ):
        """
        Parameters
        ----------
        model_dir : str
            Directory containing ONNX source files.
        engine_dir : str
            Directory for compiled TensorRT engines.
        trtexec_path : str, optional
            Explicit path to ``trtexec``.  Searched in PATH if omitted.
        force_recompile : bool
            Always recompile, even if a compatible engine exists.
        platform_prefix : str
            Optional prefix prepended to engine filenames (e.g. "thor_").
            Allows the same engine_dir to hold engines for multiple platforms.
        """
        self.model_dir = Path(model_dir)
        self.engine_dir = Path(engine_dir)
        self.force_recompile = force_recompile
        self.platform_prefix = platform_prefix

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.engine_dir.mkdir(parents=True, exist_ok=True)

        # TensorRT version
        self.trt_version = trt.__version__
        self.trt_major = int(self.trt_version.split(".")[0])
        logger.info("TensorRT %s (major=%d)", self.trt_version, self.trt_major)

        # Locate trtexec
        self.trtexec_path = self._find_trtexec(trtexec_path)

    #  Public API

    def ensure_engine(self, model_key: str) -> Path:
        """Return path to a ready-to-use engine, compiling if necessary.

        Raises
        ------
        FileNotFoundError
            If the ONNX source is missing.
        RuntimeError
            If compilation fails.
        """
        cfg = self._get_config(model_key)
        onnx_path = self.model_dir / cfg["onnx_name"]
        engine_path = self.engine_dir / (self.platform_prefix + cfg["engine_name"])

        # Fast path: compatible engine already exists
        if not self.force_recompile and self._engine_ok(engine_path):
            logger.info("Engine ready: %s", engine_path)
            return engine_path

        # Need to compile
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {onnx_path}\n"
                f"Place the file there or set --model-dir to the correct location."
            )

        # Back up old engine
        if engine_path.exists():
            bak = engine_path.with_suffix(f".bak.trt{self.trt_version}")
            logger.info("Backing up old engine → %s", bak)
            shutil.move(engine_path, bak)

        logger.info(
            "Compiling %s → %s  (this may take a few minutes...)",
            onnx_path.name,
            engine_path.name,
        )
        self._compile(cfg, onnx_path, engine_path)
        return engine_path

    def ensure_all(self, keys: Optional[List[str]] = None) -> Dict[str, Path]:
        """Ensure engines for multiple models. Returns {key: engine_path}.

        Parameters
        ----------
        keys : list[str], optional
            Model keys to prepare.  ``None`` means all registered models.
        """
        keys = keys or list(MODEL_REGISTRY.keys())
        results: Dict[str, Path] = {}
        for k in keys:
            try:
                results[k] = self.ensure_engine(k)
            except Exception as exc:
                logger.error("Failed to prepare %s: %s", k, exc)
                results[k] = None  # type: ignore[assignment]
        return results

    def list_models(self) -> None:
        """Print a summary table of registered models and their status."""
        print(f"\n{'Model':<20} {'Engine':<40} {'Status'}")
        print("-" * 80)
        for key, cfg in MODEL_REGISTRY.items():
            engine_path = self.engine_dir / (self.platform_prefix + cfg["engine_name"])
            onnx_path = self.model_dir / cfg["onnx_name"]

            if self._engine_ok(engine_path):
                status = f"✓ ready ({engine_path.stat().st_size / 1024 / 1024:.1f} MB)"
            elif onnx_path.exists():
                status = "⚠ needs compile (ONNX found)"
            else:
                status = "✗ missing ONNX"

            print(f"{key:<20} {cfg['engine_name']:<40} {status}")
        print()

    #  Internal helpers

    @staticmethod
    def _find_trtexec(explicit_path: Optional[str]) -> Path:
        """Locate the trtexec binary."""
        candidates = [
            explicit_path,
            shutil.which("trtexec"),
            "/usr/src/tensorrt/bin/trtexec",
            "/usr/local/bin/trtexec",
        ]
        for c in candidates:
            if c and Path(c).exists():
                logger.info("Using trtexec: %s", c)
                return Path(c)
        raise FileNotFoundError(
            "trtexec not found. Install TensorRT or pass --trtexec-path."
        )

    @staticmethod
    def _get_config(model_key: str) -> dict:
        if model_key not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown model '{model_key}'. "
                f"Available: {', '.join(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[model_key]

    def _engine_ok(self, engine_path: Path) -> bool:
        """Return True if engine exists and deserializes on the current TRT runtime."""
        if not engine_path.exists():
            return False
        try:
            trt_logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    logger.warning("Engine failed to deserialize: %s", engine_path)
                    return False
            return True
        except Exception as exc:
            logger.warning("Engine check failed (%s): %s", engine_path, exc)
            return False

    @staticmethod
    def _onnx_has_dynamic_dims(onnx_path: Path) -> bool:
        """Check if any input tensor in the ONNX model has dynamic dimensions."""
        try:
            import onnx

            model = onnx.load(str(onnx_path), load_external_data=False)
            for inp in model.graph.input:
                for dim in inp.type.tensor_type.shape.dim:
                    # Dynamic if dim_param is set (e.g. "batch") or dim_value is 0/-1
                    if dim.dim_param:
                        return True
                    if dim.dim_value <= 0:
                        return True
            return False
        except ImportError:
            logger.warning("onnx package not installed — assuming dynamic shapes")
            return True
        except Exception as exc:
            logger.warning(
                "Could not inspect ONNX (%s): %s — assuming dynamic",
                onnx_path.name,
                exc,
            )
            return True

    def _compile(self, cfg: dict, onnx_path: Path, engine_path: Path) -> None:
        """Run trtexec to compile ONNX → engine."""
        cmd = [
            str(self.trtexec_path),
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]

        # Workspace
        ws = cfg.get("workspace", 4096)
        if self.trt_major >= 10:
            cmd.append(f"--memPoolSize=workspace:{ws}M")
        else:
            cmd.append(f"--workspace={ws}")

        # Precision
        if cfg.get("fp16", True):
            cmd.append("--fp16")

        # Shapes — only pass if ONNX actually has dynamic dims
        has_dynamic_dims = self._onnx_has_dynamic_dims(onnx_path)
        input_name = cfg.get("input_name", "images")

        if "dynamic" in cfg and has_dynamic_dims:
            d = cfg["dynamic"]
            cmd.append(f"--minShapes={input_name}:{d['min']}")
            cmd.append(f"--optShapes={input_name}:{d['opt']}")
            cmd.append(f"--maxShapes={input_name}:{d['max']}")
        elif "fixed_shape" in cfg and has_dynamic_dims:
            shape = f"{input_name}:{cfg['fixed_shape']}"
            cmd.extend(
                [
                    f"--minShapes={shape}",
                    f"--optShapes={shape}",
                    f"--maxShapes={shape}",
                ]
            )
        elif not has_dynamic_dims:
            logger.info("ONNX has static shapes — skipping shape args")

        logger.info("Running: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            # Log last few lines
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            for line in lines[-10:]:
                logger.info("  trtexec: %s", line)
        except subprocess.CalledProcessError as exc:
            logger.error("trtexec failed (rc=%d):\n%s", exc.returncode, exc.stdout)
            raise RuntimeError(
                f"Engine compilation failed for {onnx_path.name}"
            ) from exc

        if not engine_path.exists():
            raise RuntimeError(f"trtexec completed but engine not found: {engine_path}")

        size_mb = engine_path.stat().st_size / 1024 / 1024
        logger.info("✓ Engine ready: %s (%.1f MB)", engine_path.name, size_mb)


# CLI
def main() -> None:
    """CLI entry point for model management."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "..", "model")
    default_engine_dir = os.path.join(script_dir, "..", "engine")

    ap = argparse.ArgumentParser(
        description="Manage TensorRT engines for face-recognition pipeline"
    )
    ap.add_argument(
        "--model-dir",
        default=default_model_dir,
        help="Directory with ONNX files (default: ../model relative to this script)",
    )
    ap.add_argument(
        "--engine-dir",
        default=default_engine_dir,
        help="Directory for engine files (default: ../engine relative to this script)",
    )
    ap.add_argument(
        "--platform-prefix",
        default="",
        help="Prefix for engine filenames, e.g. 'thor_'",
    )
    ap.add_argument(
        "--trtexec-path",
        default=None,
        help="Path to trtexec binary",
    )
    ap.add_argument(
        "--force-recompile",
        action="store_true",
        help="Recompile even if compatible engine exists",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List all models and their status",
    )
    ap.add_argument(
        "--prepare",
        default=None,
        help="Prepare engine(s): model key or 'all'",
    )

    args = ap.parse_args()

    mm = ModelManager(
        model_dir=args.model_dir,
        engine_dir=args.engine_dir,
        trtexec_path=args.trtexec_path,
        force_recompile=args.force_recompile,
        platform_prefix=args.platform_prefix,
    )

    if args.list:
        mm.list_models()
        return

    if args.prepare:
        if args.prepare == "all":
            results = mm.ensure_all()
            print(f"\n{'='*60}")
            print("SUMMARY:")
            for key, path in results.items():
                status = f"✓ {path}" if path else "✗ FAILED"
                print(f"  {key:<20} {status}")
            print(f"{'='*60}")
            if any(p is None for p in results.values()):
                sys.exit(1)
        else:
            engine = mm.ensure_engine(args.prepare)
            print(f"\n✓ Ready: {engine}")
        return

    # Default: just list
    mm.list_models()


if __name__ == "__main__":
    main()
