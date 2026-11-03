// Stand-in for the 'isaac_ros_common/cuda_stream.hpp' header of Isaac ROS, which is not available outside the Isaac
// ROS workspace. The FoundationPose core sources only use 'CHECK_CUDA_ERROR' from it.
#ifndef ROBOKUDO_FOUNDATION_POSE_TENSORRT__ISAAC_ROS_COMMON__CUDA_STREAM_HPP_
#define ROBOKUDO_FOUNDATION_POSE_TENSORRT__ISAAC_ROS_COMMON__CUDA_STREAM_HPP_

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define CHECK_CUDA_ERROR(code, msg) \
  do { \
    const cudaError_t rk_cuda_error_ = (code); \
    if (rk_cuda_error_ != cudaSuccess) { \
      throw std::runtime_error( \
              std::string("CUDA error (") + (msg) + "): " + cudaGetErrorString(rk_cuda_error_)); \
    } \
  } while (0)

#endif  // ROBOKUDO_FOUNDATION_POSE_TENSORRT__ISAAC_ROS_COMMON__CUDA_STREAM_HPP_
