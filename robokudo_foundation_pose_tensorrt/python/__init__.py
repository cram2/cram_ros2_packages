"""TensorRT accelerated FoundationPose (Isaac ROS core) inside the Python process, see the package README.

    from robokudo_foundation_pose_tensorrt import FoundationPose, ensure_engines
    refine_engine, score_engine = ensure_engines(precision="fp16")
    est = FoundationPose(refine_engine_file_path=refine_engine, score_engine_file_path=score_engine)
"""
import ctypes
import os.path as osp

from robokudo_foundation_pose_tensorrt.engines import engine_paths, ensure_engines


def _load_tensorrt() -> None:
    """Load libnvinfer of the 'tensorrt' wheel of the running Python environment.

    The extension does not hard code the venv it was built with, so any venv with the same TensorRT major version
    works. A library already loaded under its soname satisfies the extension's dependency.
    """
    try:
        import tensorrt_libs
    except ImportError as e:
        raise ImportError("robokudo_foundation_pose_tensorrt needs TensorRT 10 in this Python environment: "
                          "'python3 -m pip install \"tensorrt-cu12>=10.13.3,<11\"'") from e
    ctypes.CDLL(osp.join(osp.dirname(tensorrt_libs.__file__), "libnvinfer.so.10"), mode=ctypes.RTLD_GLOBAL)


_load_tensorrt()

try:
    from robokudo_foundation_pose_tensorrt._bindings import FoundationPose
except ImportError as e:
    raise ImportError("The robokudo_foundation_pose_tensorrt extension is missing or broken. 'colcon build' of "
                      "'robokudo_foundation_pose_tensorrt' skips it without a CUDA compiler or TensorRT, see its build "
                      "warnings.") from e

__all__ = ["FoundationPose", "engine_paths", "ensure_engines"]
