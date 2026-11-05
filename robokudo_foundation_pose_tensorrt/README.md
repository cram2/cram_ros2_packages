# robokudo_foundation_pose_tensorrt: in-process TensorRT FoundationPose

ROS 2 package providing the Python module `robokudo_foundation_pose_tensorrt`, which runs the Isaac ROS FoundationPose
pipeline inside the RoboKudo process, without ROS, the Isaac container or the Isaac node. It compiles the ROS-free core
of Isaac ROS FoundationPose in `isaac/` (pose sampling, nvdiffrast CUDA rendering, pose update) and calls the TensorRT
refine/score engines directly. `src/estimator.cpp` is a port of `FoundationPoseNode::processFrame`.

`FoundationPoseAnnotator` uses it.

```python
from robokudo_foundation_pose_tensorrt import FoundationPose, ensure_engines

refine_engine, score_engine = ensure_engines(precision="fp16")    # cached, built on first use
est = FoundationPose(refine_engine_file_path=refine_engine, score_engine_file_path=score_engine,
                     refine_iterations=3, min_depth=0.1, max_depth=4.0)
est.set_mesh("/abs/path/textured.obj")     # loaded once (raises on errors), cached on the GPU: later switches are free
result = est.estimate(rgb=rgb, depth=depth, mask=mask, cam_intrinsics=K)
if result is not None:
    pose, extents, score = result          # [4, 4] object to camera, [3] mesh size, best hypothesis score
    # next frame: refine the known pose instead of a full estimation (~3 ms instead of ~370 ms)
    pose, extents, _ = est.track(rgb=rgb2, depth=depth2, mask=mask2, cam_intrinsics=K, previous_pose=pose, iterations=2)
```

`rgb` [H, W, 3] uint8 RGB, `depth` [H, W] float32 meters, `mask` [H, W] uint8/bool, `K` [3, 3]. All calls release
the GIL, so they run in a `ThreadedAnnotator` while the tree ticks.

## Layout

- `python/`: the Python module `robokudo_foundation_pose_tensorrt` (`__init__.py`, `engines.py`). It is installed under
  the module name next to the compiled extension, import it only from the install space.
- `src/`: the RoboKudo part, the estimator (sample, render, refine, score, decode, track), direct TensorRT inference
  and the pybind11 bindings.
- `isaac/`: the core sources of `isaac_ros_foundationpose` (Isaac ROS Pose Estimation 5.0, Apache 2.0, see
  `isaac/LICENSE`), see `isaac/README.md`.

## Build

A normal workspace build, with the RoboKudo venv active (`rksetup`):

```bash
python3 -m pip install "tensorrt-cu12>=10.13.3,<11"    # once per venv, provides the TensorRT runtime
python3 -m colcon build --symlink-install --packages-select robokudo_foundation_pose_tensorrt
```

- Needs a CUDA compiler (`nvidia-cuda-toolkit`; with nvcc < 12.4, `g++-12` is used as host compiler automatically)
  plus Eigen, assimp and OpenCV (`rosdep install --from-paths src/cram_ros2_packages/robokudo_foundation_pose_tensorrt`).
  Without a CUDA compiler or TensorRT in the build venv, only the Python part is installed and the build warns.
- pybind11 2.13 (NumPy 2 support), CV-CUDA 0.19 and the TensorRT 10.13.3 headers (not part of the wheel) are
  downloaded and checksum verified at configure time.
- Builds in Release and for the GPU of the build machine (`CMAKE_CUDA_ARCHITECTURES=native`).
- The extension does not hard code the venv: `__init__.py` loads `libnvinfer` from the `tensorrt` wheel of
  the running Python, so every venv with that wheel works.

### Engines

Engines only run with the TensorRT version and GPU they were built for. They are built from the ONNX models
(downloaded from NGC, checksum verified) into `~/.cache/robokudo_foundation_pose_tensorrt/<tensorrt>_<gpu>/` (or
`$ROBOKUDO_FOUNDATION_POSE_TENSORRT_CACHE`), with the optimization profiles tuned for the batch sizes used at runtime (refine 42,
score 252). The colcon build creates the FP16 engines if they are missing (a few minutes, once per GPU), the
annotator builds missing ones on first use, and `python3 python/engines.py --precision fp32` builds the FP32 ones.

## Tests

```bash
python3 -m colcon test --packages-select robokudo_foundation_pose_tensorrt && python3 -m colcon test-result --verbose
```

Generates textured boxes, renders them at known poses and checks estimation, tracking, mesh cache switches and error
handling against the ground truth.

## Performance (RTX 4070, 640x480, synthetic scenes)

| engines | call | time | rotation / translation error |
|---|---|---|---|
| FP32, opt shape 1 (Isaac docs) | estimate, 3 refine iterations | ~1470 ms | 0.2-0.6 deg / 0.7-1.4 mm |
| FP16, opt shape 1 | estimate, 3 refine iterations | ~423 ms | 0.2-1.0 deg / 0.6-1.3 mm |
| **FP16, opt shape 42/252 (default)** | estimate, 3 refine iterations | **~370 ms** | 0.1-0.9 deg / 0.6-1.4 mm |
| FP16, opt shape 42/252 | estimate, 1 refine iteration | ~210 ms | 0.9-5.5 deg / 2.5-20 mm |
| FP16, opt shape 42/252 | track, 2 iterations | ~3 ms | 0.6 deg / 0.1 mm after a 4 deg / 39 mm motion |
