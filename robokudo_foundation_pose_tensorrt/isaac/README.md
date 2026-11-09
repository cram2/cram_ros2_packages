# Isaac ROS FoundationPose core

The ROS-free core of the `isaac_ros_foundationpose` package of
[Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation) release 5.0 (Apache 2.0,
see `LICENSE`), compiled into `robokudo_foundation_pose_tensorrt`: pose sampling, pose rendering with nvdiffrast, pose
transformation and their CUDA kernels. `foundationpose_model_config.yaml` holds the model config values that
`src/estimator.hpp` uses.

Not taken over: the ROS nodes, launch files and the model install package. `pose_decoder` depends on `vision_msgs`, its
decode step is reimplemented in `src/estimator.cpp`.

Changes to the upstream sources:

- `src/foundationpose_impl/mesh_loader.cpp` computes the mesh diameter on at most 5000 vertices instead of all vertex
  pairs, which otherwise takes seconds for dense scanned meshes.
- `src/foundationpose_impl/pose_renderer.cpp` keys its texture cache by a unique texture id instead of the vertex
  count, which mixed up textures of different meshes.
- `include/isaac_ros_common/cuda_stream.hpp` is a stand-in for the header of `isaac_ros_common`. It only provides
  `CHECK_CUDA_ERROR`, which throws a C++ exception.
