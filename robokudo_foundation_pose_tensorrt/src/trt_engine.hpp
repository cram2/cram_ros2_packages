// Minimal TensorRT runtime wrapper: load a serialized engine and run it on caller owned device buffers.
#ifndef ROBOKUDO_FOUNDATION_POSE_TENSORRT__TRT_ENGINE_HPP_
#define ROBOKUDO_FOUNDATION_POSE_TENSORRT__TRT_ENGINE_HPP_

#include <cuda_runtime.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "NvInferRuntime.h"

namespace robokudo_foundation_pose_tensorrt
{

class TrtEngine
{
public:
  explicit TrtEngine(const std::string & engine_file_path);
  ~TrtEngine();

  TrtEngine(const TrtEngine &) = delete;
  TrtEngine & operator=(const TrtEngine &) = delete;

  // Enqueue one inference on 'stream'. 'inputs' maps input names to (device pointer, shape), 'outputs' maps output
  // names to device pointers large enough for the shapes resulting from the inputs.
  void enqueue(
    const std::map<std::string, std::pair<const float *, std::vector<int64_t>>> & inputs,
    const std::map<std::string, float *> & outputs, cudaStream_t stream);

  // Largest batch size of the optimization profile of input 'name'.
  int64_t maxBatchSize(const std::string & name) const;

private:
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;
  std::string engine_file_path_;
};

}  // namespace robokudo_foundation_pose_tensorrt

#endif  // ROBOKUDO_FOUNDATION_POSE_TENSORRT__TRT_ENGINE_HPP_
