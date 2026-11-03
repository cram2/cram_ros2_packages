#include "trt_engine.hpp"

#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace robokudo_foundation_pose_tensorrt
{

namespace
{
class Logger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[robokudo_foundation_pose_tensorrt TensorRT] " << msg << std::endl;
    }
  }
};

Logger & logger()
{
  static Logger instance;
  return instance;
}
}  // namespace

TrtEngine::TrtEngine(const std::string & engine_file_path)
: engine_file_path_(engine_file_path)
{
  std::ifstream file(engine_file_path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Cannot open TensorRT engine: " + engine_file_path);
  }
  const std::vector<char> data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

  runtime_.reset(nvinfer1::createInferRuntime(logger()));
  if (!runtime_) {
    throw std::runtime_error("Cannot create the TensorRT runtime");
  }
  engine_.reset(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    throw std::runtime_error(
            "Cannot deserialize TensorRT engine '" + engine_file_path +
            "' (built with another TensorRT version or for another GPU?)");
  }
  context_.reset(engine_->createExecutionContext());
  if (!context_) {
    throw std::runtime_error("Cannot create a TensorRT execution context for " + engine_file_path);
  }
}

TrtEngine::~TrtEngine()
{
  // destroy in reverse order of creation
  context_.reset();
  engine_.reset();
  runtime_.reset();
}

void TrtEngine::enqueue(
  const std::map<std::string, std::pair<const float *, std::vector<int64_t>>> & inputs,
  const std::map<std::string, float *> & outputs, cudaStream_t stream)
{
  for (const auto & [name, input] : inputs) {
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int32_t>(input.second.size());
    for (int32_t i = 0; i < dims.nbDims; ++i) {
      dims.d[i] = input.second[i];
    }
    if (!context_->setInputShape(name.c_str(), dims) ||
      !context_->setTensorAddress(name.c_str(), const_cast<float *>(input.first)))
    {
      throw std::runtime_error("Invalid TensorRT input '" + name + "' for " + engine_file_path_);
    }
  }
  for (const auto & [name, ptr] : outputs) {
    if (!context_->setTensorAddress(name.c_str(), ptr)) {
      throw std::runtime_error("Invalid TensorRT output '" + name + "' for " + engine_file_path_);
    }
  }
  if (!context_->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT inference failed for " + engine_file_path_);
  }
}

int64_t TrtEngine::maxBatchSize(const std::string & name) const
{
  return engine_->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX).d[0];
}

}  // namespace robokudo_foundation_pose_tensorrt
