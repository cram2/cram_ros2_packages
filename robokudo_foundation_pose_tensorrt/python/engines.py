"""TensorRT engines of the FoundationPose refine and score networks.

Engines only run with the TensorRT version and GPU they were built for, so they are built on demand from the ONNX
models into a per user cache, '~/.cache/robokudo_foundation_pose_tensorrt/<tensorrt version>_<gpu>/' (or
'$ROBOKUDO_FOUNDATION_POSE_TENSORRT_CACHE/...').
The optimization profiles are tuned for the batch sizes used at runtime (refine 42, score 252), not for batch 1 as
in the Isaac ROS docs, which is ~13 % faster.

Usable without the compiled module, also as a script: 'python3 engines.py [--precision fp16|fp32]'.
"""
import argparse
import hashlib
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_NGC_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/isaac/foundationpose/versions/1.0.1_onnx/files/"
# name: (sha256 of the ONNX model, max batch of the engine)
_MODELS: Dict[str, Tuple[str, int]] = {
    "refine": ("dcc695a19c4bcfe5e1d909a22d8f652d8ec8bab1e19bd1544c6b45f2d3595cf7", 42),
    "score": ("0bf1026c0db7320ebf9a548ecf0d3c810c8dbd377948630bd3e5af1d49440503", 252),
}
_INPUT_SHAPE = (160, 160, 6)


def cache_root() -> str:
    """Root directory of the engine and model cache."""
    return os.environ.get("ROBOKUDO_FOUNDATION_POSE_TENSORRT_CACHE",
                          osp.join(osp.expanduser("~"), ".cache", "robokudo_foundation_pose_tensorrt"))


def _gpu_tag() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown_gpu"
    return re.sub(r"[^A-Za-z0-9.]+", "_", out).strip("_")


def engine_dir() -> str:
    """Directory of the engines for the installed TensorRT version and the current GPU."""
    import tensorrt as trt
    return osp.join(cache_root(), f"trt{trt.__version__}_{_gpu_tag()}")


def engine_paths(precision: str = "fp16") -> Tuple[str, str]:
    """(refine, score) engine paths for 'precision', they may not exist yet, see 'ensure_engines'."""
    if precision not in ("fp16", "fp32"):
        raise ValueError(f"Unknown precision '{precision}', use 'fp16' or 'fp32'.")
    directory = engine_dir()
    return tuple(osp.join(directory, f"{name}_{precision}.plan") for name in ("refine", "score"))


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _onnx_model(name: str) -> str:
    """Path of the downloaded and verified ONNX model 'name'."""
    expected_sha256 = _MODELS[name][0]
    path = osp.join(cache_root(), "onnx", f"{name}_model.onnx")
    if osp.isfile(path) and _sha256(path) == expected_sha256:
        return path

    os.makedirs(osp.dirname(path), exist_ok=True)
    logger.warning(f"Downloading the FoundationPose {name} model from NGC")
    with tempfile.NamedTemporaryFile(dir=osp.dirname(path), delete=False) as tmp:
        with urllib.request.urlopen(f"{_NGC_URL}{name}_model.onnx", timeout=60) as response:
            shutil.copyfileobj(response, tmp)
    if _sha256(tmp.name) != expected_sha256:
        os.remove(tmp.name)
        raise RuntimeError(f"Checksum mismatch of the downloaded FoundationPose {name} model")
    os.replace(tmp.name, path)
    return path


def _build_engine(name: str, precision: str, path: str) -> None:
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(_onnx_model(name)):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    batch = _MODELS[name][1]
    profile = builder.create_optimization_profile()
    for tensor in ("input1", "input2"):
        profile.set_shape(tensor, (1, *_INPUT_SHAPE), (batch, *_INPUT_SHAPE), (batch, *_INPUT_SHAPE))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"Building the FoundationPose {name} engine failed")
    os.makedirs(osp.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=osp.dirname(path), delete=False) as tmp:
        tmp.write(engine)
    os.replace(tmp.name, path)


def ensure_engines(precision: str = "fp16") -> Tuple[str, str]:
    """(refine, score) engine paths, building missing engines first (takes a few minutes, downloads the models).

    :param str precision: 'fp16' (fast) or 'fp32' (what the Isaac ROS docs recommend).
    :rtype: Tuple[str, str]
    """
    paths = engine_paths(precision)
    for name, path in zip(("refine", "score"), paths):
        if not osp.isfile(path):
            logger.warning(f"Building the FoundationPose {name} {precision} TensorRT engine '{path}', this takes "
                           f"a few minutes")
            _build_engine(name, precision, path)
    return paths


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--no-fail", action="store_true",
                        help="only warn on errors (used by the colcon build, the annotator retries at runtime)")
    args = parser.parse_args(argv)
    # stdout: colcon reports any stderr output of a build as a problem
    logging.basicConfig(level=logging.INFO, format="[robokudo_foundation_pose_tensorrt] %(message)s", stream=sys.stdout)
    try:
        for path in ensure_engines(args.precision):
            logger.info(f"engine: {path}")
    except Exception as e:  # e.g. no network, no GPU or no 'tensorrt' wheel
        if not args.no_fail:
            raise
        logger.warning(f"Could not build the FoundationPose engines ({e}), they are built on first use.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
