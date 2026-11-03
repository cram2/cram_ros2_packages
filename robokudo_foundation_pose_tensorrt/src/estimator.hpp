// FoundationPose pose estimation in-process: port of 'FoundationPoseNode::processFrame' of Isaac ROS FoundationPose
// with direct TensorRT inference and without ROS.
#ifndef ROBOKUDO_FOUNDATION_POSE_TENSORRT__ESTIMATOR_HPP_
#define ROBOKUDO_FOUNDATION_POSE_TENSORRT__ESTIMATOR_HPP_

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_renderer.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_sampler.hpp"
#include "isaac_ros_foundationpose/foundationpose_impl/pose_transformer.hpp"
#include "trt_engine.hpp"

namespace robokudo_foundation_pose_tensorrt
{

namespace fp = nvidia::isaac_ros::foundationpose;

struct EstimatorParams
{
  std::string refine_engine_file_path;
  std::string score_engine_file_path;
  int refine_iterations{3};
  float min_depth{0.1f};
  float max_depth{4.0f};
  // values of config/foundationpose_model_config.yaml of Isaac ROS FoundationPose, they belong to the models
  uint32_t max_hypothesis{252};
  uint32_t resized_image_height{160};
  uint32_t resized_image_width{160};
  float refine_crop_ratio{1.2f};
  float score_crop_ratio{1.2f};
  float rot_normalizer{0.349065850398865f};
  std::vector<std::string> symmetry_axes;
  std::vector<std::string> fixed_axis_angles;
  std::vector<std::string> fixed_translations;
};

struct EstimateResult
{
  Eigen::Matrix4f pose;         // object (original mesh origin) to camera
  Eigen::Vector3f extents;      // axis aligned size of the mesh
  float score{0.0f};            // score of the selected hypothesis
};

class Estimator
{
public:
  explicit Estimator(const EstimatorParams & params);
  ~Estimator();

  Estimator(const Estimator &) = delete;
  Estimator & operator=(const Estimator &) = delete;

  // Select the mesh for the following 'estimate' calls. A mesh is loaded to the GPU once (synchronously, throws on
  // errors) and stays cached, so switching back to it is free.
  void setMesh(const std::string & mesh_file_path);
  // Path of the selected mesh, empty if none.
  std::string meshFilePath() const;
  // Drop all cached meshes except the selected one.
  void clearMeshCache();

  // Host images: rgb [h, w, 3] uint8, depth [h, w] float meters, mask [h, w] uint8 (non zero = object).
  // Returns no value if no pose hypothesis could be sampled (e.g. empty mask or no valid depth).
  std::optional<EstimateResult> estimate(
    const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
    const Eigen::Matrix3f & cam_intrinsics);

  // Refine a known pose (e.g. of the previous frame) with 'iterations' refine passes, without sampling and scoring.
  // 'previous_pose' and the result map the original mesh origin to the camera. Much faster than 'estimate'.
  EstimateResult track(
    const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
    const Eigen::Matrix3f & cam_intrinsics, const Eigen::Matrix4f & previous_pose, int iterations);

private:
  std::shared_ptr<const fp::MeshData> currentMeshData() const;
  fp::FrameObservationView uploadFrame(
    const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
    const Eigen::Matrix3f & cam_intrinsics);

  void allocateFrameBuffers(const fp::ImageDim & size);
  void freeFrameBuffers();

  EstimatorParams params_;
  mutable std::mutex mutex_;
  cudaStream_t stream_{nullptr};

  std::unique_ptr<TrtEngine> refine_engine_;
  std::unique_ptr<TrtEngine> score_engine_;
  std::map<std::string, std::unique_ptr<fp::MeshLoader>> meshes_;
  fp::MeshLoader * current_mesh_{nullptr};
  std::unique_ptr<fp::PoseSampler> sampler_;
  std::unique_ptr<fp::PoseRenderer> refine_renderer_;
  std::unique_ptr<fp::PoseRenderer> score_renderer_;
  std::unique_ptr<fp::PoseRenderer> track_renderer_;
  std::unique_ptr<fp::PoseTransformer> transformer_;

  // per frame size
  fp::ImageDim frame_size_;
  uint8_t * rgb_gpu_{nullptr};
  float * depth_gpu_{nullptr};
  uint8_t * mask_gpu_{nullptr};
  float * pc_gpu_{nullptr};

  // per max_hypothesis
  int64_t refine_max_batch_{0};
  float * poses_gpu_{nullptr};
  float * refine_rendered_gpu_{nullptr};
  float * refine_observed_gpu_{nullptr};
  float * refine_trans_gpu_{nullptr};
  float * refine_rot_gpu_{nullptr};
  float * score_rendered_gpu_{nullptr};
  float * score_observed_gpu_{nullptr};
  float * scores_gpu_{nullptr};
};

}  // namespace robokudo_foundation_pose_tensorrt

#endif  // ROBOKUDO_FOUNDATION_POSE_TENSORRT__ESTIMATOR_HPP_
