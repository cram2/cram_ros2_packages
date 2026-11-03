#include "estimator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

#include "foundationpose_sampling.cu.hpp"
#include "isaac_ros_common/cuda_stream.hpp"

namespace robokudo_foundation_pose_tensorrt
{

namespace
{
constexpr int32_t kChannels = 6;  // rgb + xyz per pixel of the rendered/observed network inputs

template<typename T>
void cudaFreeAndReset(T *& ptr)
{
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template<typename T>
void cudaAlloc(T *& ptr, size_t count, const char * what)
{
  CHECK_CUDA_ERROR(cudaMalloc(&ptr, count * sizeof(T)), what);
}
}  // namespace

Estimator::Estimator(const EstimatorParams & params)
: params_(params)
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_), "create stream");

  refine_engine_ = std::make_unique<TrtEngine>(params_.refine_engine_file_path);
  score_engine_ = std::make_unique<TrtEngine>(params_.score_engine_file_path);

  fp::PoseSamplerParams sampler_params;
  sampler_params.max_hypothesis = params_.max_hypothesis;
  sampler_params.min_depth = params_.min_depth;
  sampler_params.symmetry_axes = params_.symmetry_axes;
  sampler_params.fixed_axis_angles = params_.fixed_axis_angles;
  sampler_params.fixed_translations = params_.fixed_translations;
  sampler_ = std::make_unique<fp::PoseSampler>(sampler_params, stream_);

  fp::PoseRendererParams render_params;
  render_params.crop_ratio = params_.refine_crop_ratio;
  render_params.min_depth = params_.min_depth;
  render_params.max_depth = params_.max_depth;
  render_params.resized_height = params_.resized_image_height;
  render_params.resized_width = params_.resized_image_width;
  refine_renderer_ = std::make_unique<fp::PoseRenderer>(render_params, stream_);

  fp::PoseRendererParams score_render_params = render_params;
  score_render_params.crop_ratio = params_.score_crop_ratio;
  score_renderer_ = std::make_unique<fp::PoseRenderer>(score_render_params, stream_);
  // separate renderer for single pose tracking, the renderers reallocate on every change of the pose count
  track_renderer_ = std::make_unique<fp::PoseRenderer>(render_params, stream_);

  transformer_ = std::make_unique<fp::PoseTransformer>(params_.rot_normalizer, stream_);

  // refine runs in chunks of the largest batch the refine engine supports
  refine_max_batch_ = std::min<int64_t>(refine_engine_->maxBatchSize("input1"), params_.max_hypothesis);
  const int64_t score_max_batch = score_engine_->maxBatchSize("input1");
  if (score_max_batch < params_.max_hypothesis) {
    throw std::runtime_error(
            "Score engine max batch size " + std::to_string(score_max_batch) + " < max_hypothesis " +
            std::to_string(params_.max_hypothesis));
  }

  const size_t floats_per_pose =
    static_cast<size_t>(params_.resized_image_height) * params_.resized_image_width * kChannels;
  cudaAlloc(poses_gpu_, params_.max_hypothesis * 16, "malloc poses");
  cudaAlloc(refine_rendered_gpu_, refine_max_batch_ * floats_per_pose, "malloc refine rendered");
  cudaAlloc(refine_observed_gpu_, refine_max_batch_ * floats_per_pose, "malloc refine observed");
  cudaAlloc(refine_trans_gpu_, refine_max_batch_ * 3, "malloc refine trans");
  cudaAlloc(refine_rot_gpu_, refine_max_batch_ * 3, "malloc refine rot");
  cudaAlloc(score_rendered_gpu_, params_.max_hypothesis * floats_per_pose, "malloc score rendered");
  cudaAlloc(score_observed_gpu_, params_.max_hypothesis * floats_per_pose, "malloc score observed");
  cudaAlloc(scores_gpu_, params_.max_hypothesis, "malloc scores");
}

Estimator::~Estimator()
{
  if (stream_) {
    cudaStreamSynchronize(stream_);
  }
  refine_renderer_.reset();
  score_renderer_.reset();
  track_renderer_.reset();
  sampler_.reset();
  transformer_.reset();
  current_mesh_ = nullptr;
  meshes_.clear();
  refine_engine_.reset();
  score_engine_.reset();
  freeFrameBuffers();
  cudaFreeAndReset(poses_gpu_);
  cudaFreeAndReset(refine_rendered_gpu_);
  cudaFreeAndReset(refine_observed_gpu_);
  cudaFreeAndReset(refine_trans_gpu_);
  cudaFreeAndReset(refine_rot_gpu_);
  cudaFreeAndReset(score_rendered_gpu_);
  cudaFreeAndReset(score_observed_gpu_);
  cudaFreeAndReset(scores_gpu_);
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

void Estimator::setMesh(const std::string & mesh_file_path)
{
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = meshes_.find(mesh_file_path);
  if (it == meshes_.end()) {
    auto loader = std::make_unique<fp::MeshLoader>(stream_);
    loader->load(mesh_file_path);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync mesh upload");
    it = meshes_.emplace(mesh_file_path, std::move(loader)).first;
  }
  current_mesh_ = it->second.get();
}

std::string Estimator::meshFilePath() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return current_mesh_ ? current_mesh_->getMeshData()->mesh_file_path : std::string();
}

void Estimator::clearMeshCache()
{
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync before clearing meshes");
  for (auto it = meshes_.begin(); it != meshes_.end(); ) {
    it = it->second.get() == current_mesh_ ? std::next(it) : meshes_.erase(it);
  }
}

void Estimator::allocateFrameBuffers(const fp::ImageDim & size)
{
  if (rgb_gpu_ && frame_size_ == size) {
    return;
  }
  freeFrameBuffers();
  cudaAlloc(rgb_gpu_, size.numPixels() * 3, "malloc rgb");
  cudaAlloc(depth_gpu_, size.numPixels(), "malloc depth");
  cudaAlloc(mask_gpu_, size.numPixels(), "malloc mask");
  cudaAlloc(pc_gpu_, size.numPixels() * 3, "malloc point cloud");
  frame_size_ = size;
}

void Estimator::freeFrameBuffers()
{
  cudaFreeAndReset(rgb_gpu_);
  cudaFreeAndReset(depth_gpu_);
  cudaFreeAndReset(mask_gpu_);
  cudaFreeAndReset(pc_gpu_);
}

std::shared_ptr<const fp::MeshData> Estimator::currentMeshData() const
{
  auto mesh_data = current_mesh_ ? current_mesh_->getMeshData() : nullptr;
  if (!mesh_data || mesh_data->num_vertices == 0) {
    throw std::runtime_error("No mesh loaded, call set_mesh first");
  }
  return mesh_data;
}

fp::FrameObservationView Estimator::uploadFrame(
  const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
  const Eigen::Matrix3f & cam_intrinsics)
{
  const fp::ImageDim frame_size{height, width};
  allocateFrameBuffers(frame_size);
  const size_t num_pixels = frame_size.numPixels();
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(rgb_gpu_, rgb, num_pixels * 3, cudaMemcpyHostToDevice, stream_), "h2d rgb");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(depth_gpu_, depth, num_pixels * sizeof(float), cudaMemcpyHostToDevice, stream_),
    "h2d depth");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(mask_gpu_, mask, num_pixels, cudaMemcpyHostToDevice, stream_), "h2d mask");

  const fp::FrameObservationView frame{
    {rgb_gpu_, frame_size, static_cast<size_t>(width) * 3, 3},
    {depth_gpu_, frame_size, static_cast<size_t>(width) * sizeof(float), 1},
    {mask_gpu_, frame_size, static_cast<size_t>(width), 1},
    {pc_gpu_, frame_size, static_cast<size_t>(width) * 3 * sizeof(float), 3},
    {cam_intrinsics(0, 0), cam_intrinsics(1, 1), cam_intrinsics(0, 2), cam_intrinsics(1, 2)}};
  if (frame.intrinsics.fx <= 0.0f || frame.intrinsics.fy <= 0.0f) {
    throw std::invalid_argument("Invalid camera intrinsics");
  }

  nvidia::isaac_ros::depth_to_xyz_map(stream_, frame.depth, frame.point_cloud, frame.intrinsics);
  CHECK_CUDA_ERROR(cudaGetLastError(), "depth_to_xyz_map");
  return frame;
}

std::optional<EstimateResult> Estimator::estimate(
  const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
  const Eigen::Matrix3f & cam_intrinsics)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto mesh_data = currentMeshData();
  const auto frame = uploadFrame(rgb, depth, mask, height, width, cam_intrinsics);

  const auto sampling = sampler_->sample(frame, mesh_data);
  if (sampling.total_poses == 0) {
    return std::nullopt;
  }
  if (static_cast<uint32_t>(sampling.total_poses) > params_.max_hypothesis) {
    throw std::runtime_error(
            "Sampled " + std::to_string(sampling.total_poses) + " poses, more than max_hypothesis " +
            std::to_string(params_.max_hypothesis));
  }
  const fp::MeshGpuView mesh_view = fp::MeshGpuView::from(*mesh_data);

  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      poses_gpu_, sampling.poses.data(), sampling.total_poses * 16 * sizeof(float),
      cudaMemcpyHostToDevice, stream_), "h2d poses");

  const uint32_t H = params_.resized_image_height;
  const uint32_t W = params_.resized_image_width;
  const auto total_poses = static_cast<uint32_t>(sampling.total_poses);

  // refine: render and update the hypotheses in chunks as large as the refine engine allows (the Isaac node uses
  // 6 batches of 42, one chunk of all hypotheses keeps the GPU busier); everything stays on the GPU and one stream
  const auto chunk_size = static_cast<uint32_t>(refine_max_batch_);
  for (int iter = 0; iter < params_.refine_iterations; ++iter) {
    for (uint32_t offset = 0; offset < total_poses; offset += chunk_size) {
      const uint32_t count = std::min(chunk_size, total_poses - offset);
      float * chunk_ptr = poses_gpu_ + static_cast<size_t>(offset) * 16;
      refine_renderer_->renderRefine(
        {chunk_ptr, count}, frame, mesh_view, {refine_rendered_gpu_, refine_observed_gpu_, count, {H, W}});

      const std::vector<int64_t> shape = {count, H, W, kChannels};
      refine_engine_->enqueue(
        {{"input1", {refine_rendered_gpu_, shape}}, {"input2", {refine_observed_gpu_, shape}}},
        {{"output1", refine_trans_gpu_}, {"output2", refine_rot_gpu_}}, stream_);

      transformer_->applyDeltas({chunk_ptr, count}, {refine_trans_gpu_, refine_rot_gpu_, count}, mesh_view);
    }
  }

  // score all refined hypotheses in one render pass and one inference
  score_renderer_->renderRefine(
    {poses_gpu_, total_poses}, frame, mesh_view, {score_rendered_gpu_, score_observed_gpu_, total_poses, {H, W}});
  const std::vector<int64_t> score_shape = {total_poses, H, W, kChannels};
  score_engine_->enqueue(
    {{"input1", {score_rendered_gpu_, score_shape}}, {"input2", {score_observed_gpu_, score_shape}}},
    {{"output1", scores_gpu_}}, stream_);

  // decode, as 'PoseDecoder::decode': best scored pose of the centered mesh, moved back to the mesh origin
  std::vector<float> scores(total_poses);
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      scores.data(), scores_gpu_, total_poses * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    "d2h scores");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync scores");
  const auto best = static_cast<size_t>(std::max_element(scores.begin(), scores.end()) - scores.begin());

  Eigen::Matrix4f pose_centered;    // column major, as the poses on the GPU
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      pose_centered.data(), poses_gpu_ + best * 16, 16 * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    "d2h best pose");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync best pose");

  Eigen::Matrix4f tf_to_center = Eigen::Matrix4f::Identity();
  tf_to_center.block<3, 1>(0, 3) = -mesh_data->mesh_model_center;

  EstimateResult result;
  result.pose = pose_centered * tf_to_center;
  result.extents = (mesh_data->max_vertex - mesh_data->min_vertex).cwiseAbs();
  result.score = scores[best];
  return result;
}

EstimateResult Estimator::track(
  const uint8_t * rgb, const float * depth, const uint8_t * mask, uint32_t height, uint32_t width,
  const Eigen::Matrix3f & cam_intrinsics, const Eigen::Matrix4f & previous_pose, int iterations)
{
  std::lock_guard<std::mutex> lock(mutex_);
  const auto mesh_data = currentMeshData();
  const auto frame = uploadFrame(rgb, depth, mask, height, width, cam_intrinsics);
  const fp::MeshGpuView mesh_view = fp::MeshGpuView::from(*mesh_data);

  // the networks work on the centered mesh, 'previous_pose' and the result belong to the original mesh origin
  Eigen::Matrix4f to_center = Eigen::Matrix4f::Identity();
  to_center.block<3, 1>(0, 3) = mesh_data->mesh_model_center;
  Eigen::Matrix4f pose_centered = previous_pose * to_center;   // column major, as the poses on the GPU
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(poses_gpu_, pose_centered.data(), 16 * sizeof(float), cudaMemcpyHostToDevice, stream_),
    "h2d track pose");

  // as the Isaac tracking node: refine the single previous pose, no sampling and no scoring
  const uint32_t H = params_.resized_image_height;
  const uint32_t W = params_.resized_image_width;
  const std::vector<int64_t> shape = {1, H, W, kChannels};
  for (int iter = 0; iter < std::max(1, iterations); ++iter) {
    track_renderer_->renderRefine(
      {poses_gpu_, 1}, frame, mesh_view, {refine_rendered_gpu_, refine_observed_gpu_, 1, {H, W}});
    refine_engine_->enqueue(
      {{"input1", {refine_rendered_gpu_, shape}}, {"input2", {refine_observed_gpu_, shape}}},
      {{"output1", refine_trans_gpu_}, {"output2", refine_rot_gpu_}}, stream_);
    transformer_->applyDeltas({poses_gpu_, 1}, {refine_trans_gpu_, refine_rot_gpu_, 1}, mesh_view);
  }

  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(pose_centered.data(), poses_gpu_, 16 * sizeof(float), cudaMemcpyDeviceToHost, stream_),
    "d2h track pose");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync track pose");

  EstimateResult result;
  result.pose = pose_centered * to_center.inverse();
  result.extents = (mesh_data->max_vertex - mesh_data->min_vertex).cwiseAbs();
  result.score = std::numeric_limits<float>::quiet_NaN();
  return result;
}

}  // namespace robokudo_foundation_pose_tensorrt
