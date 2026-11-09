// Python bindings: 'robokudo_foundation_pose_tensorrt.FoundationPose'. All heavy calls release the GIL, so the behavior
// tree keeps ticking.
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "estimator.hpp"

namespace py = pybind11;
namespace rkfp = robokudo_foundation_pose_tensorrt;

namespace
{
using RgbArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using DepthArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using MaskArray = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using RowMatrix4f = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

std::pair<uint32_t, uint32_t> checkImages(const RgbArray & rgb, const DepthArray & depth, const MaskArray & mask)
{
  if (rgb.ndim() != 3 || rgb.shape(2) != 3) {
    throw py::value_error("rgb must have the shape [H, W, 3]");
  }
  const auto height = static_cast<uint32_t>(rgb.shape(0));
  const auto width = static_cast<uint32_t>(rgb.shape(1));
  if (depth.ndim() != 2 || depth.shape(0) != height || depth.shape(1) != width) {
    throw py::value_error("depth must have the shape [H, W] of rgb");
  }
  if (mask.ndim() != 2 || mask.shape(0) != height || mask.shape(1) != width) {
    throw py::value_error("mask must have the shape [H, W] of rgb");
  }
  return {height, width};
}

std::tuple<RowMatrix4f, Eigen::Vector3f, float> track(
  rkfp::Estimator & self, const RgbArray & rgb, const DepthArray & depth, const MaskArray & mask,
  const Eigen::Matrix3f & cam_intrinsics, const RowMatrix4f & previous_pose, int iterations)
{
  const auto [height, width] = checkImages(rgb, depth, mask);
  const Eigen::Matrix4f pose = previous_pose;
  rkfp::EstimateResult result;
  {
    py::gil_scoped_release release;
    result = self.track(
      rgb.data(), depth.data(), mask.data(), height, width, cam_intrinsics, pose, iterations);
  }
  return std::make_tuple(RowMatrix4f(result.pose), result.extents, result.score);
}

std::optional<std::tuple<RowMatrix4f, Eigen::Vector3f, float>> estimate(
  rkfp::Estimator & self, const RgbArray & rgb, const DepthArray & depth, const MaskArray & mask,
  const Eigen::Matrix3f & cam_intrinsics)
{
  const auto [height, width] = checkImages(rgb, depth, mask);

  std::optional<rkfp::EstimateResult> result;
  {
    py::gil_scoped_release release;
    result = self.estimate(rgb.data(), depth.data(), mask.data(), height, width, cam_intrinsics);
  }
  if (!result) {
    return std::nullopt;
  }
  return std::make_tuple(RowMatrix4f(result->pose), result->extents, result->score);
}
}  // namespace

PYBIND11_MODULE(_bindings, m)
{
  m.doc() = "TensorRT accelerated FoundationPose (Isaac ROS core) without ROS";

  py::class_<rkfp::Estimator>(m, "FoundationPose")
  .def(
    py::init(
      [](const std::string & refine_engine_file_path, const std::string & score_engine_file_path,
      int refine_iterations, float min_depth, float max_depth,
      const std::vector<std::string> & symmetry_axes, const std::vector<std::string> & fixed_axis_angles,
      const std::vector<std::string> & fixed_translations) {
        rkfp::EstimatorParams params;
        params.refine_engine_file_path = refine_engine_file_path;
        params.score_engine_file_path = score_engine_file_path;
        params.refine_iterations = refine_iterations;
        params.min_depth = min_depth;
        params.max_depth = max_depth;
        params.symmetry_axes = symmetry_axes;
        params.fixed_axis_angles = fixed_axis_angles;
        params.fixed_translations = fixed_translations;
        py::gil_scoped_release release;
        return std::make_unique<rkfp::Estimator>(params);
      }),
    py::arg("refine_engine_file_path"), py::arg("score_engine_file_path"), py::arg("refine_iterations") = 3,
    py::arg("min_depth") = 0.1f, py::arg("max_depth") = 4.0f,
    py::arg("symmetry_axes") = std::vector<std::string>{},
    py::arg("fixed_axis_angles") = std::vector<std::string>{},
    py::arg("fixed_translations") = std::vector<std::string>{})
  .def(
    "set_mesh", &rkfp::Estimator::setMesh, py::arg("mesh_file_path"),
    py::call_guard<py::gil_scoped_release>(),
    "Select the mesh (.obj etc., in meters) for 'estimate'. It is loaded synchronously the first time (raises on "
    "errors) and cached on the GPU, switching back to it is free.")
  .def_property_readonly("mesh_file_path", &rkfp::Estimator::meshFilePath)
  .def(
    "clear_mesh_cache", &rkfp::Estimator::clearMeshCache, py::call_guard<py::gil_scoped_release>(),
    "Free all cached meshes except the selected one.")
  .def(
    "estimate", &estimate, py::arg("rgb"), py::arg("depth"), py::arg("mask"), py::arg("cam_intrinsics"),
    "Estimate the pose of the loaded mesh.\n\n"
    "rgb: [H, W, 3] uint8, depth: [H, W] float32 meters, mask: [H, W] uint8/bool, cam_intrinsics: [3, 3].\n"
    "Returns ([4, 4] object to camera pose, [3] mesh extents, score) or None if no hypothesis was sampled.")
  .def(
    "track", &track, py::arg("rgb"), py::arg("depth"), py::arg("mask"), py::arg("cam_intrinsics"),
    py::arg("previous_pose"), py::arg("iterations") = 2,
    "Refine a known [4, 4] object to camera pose (e.g. of the previous frame), no hypothesis sampling or scoring.\n"
    "Much faster than 'estimate', but only valid for small motions. Returns (pose, extents, nan).");
}
