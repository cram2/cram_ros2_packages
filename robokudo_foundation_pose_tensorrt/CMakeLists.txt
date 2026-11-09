# robokudo_foundation_pose_tensorrt: in-process TensorRT FoundationPose for RoboKudo. Python bindings around the ROS-free
# core of Isaac ROS FoundationPose in 'isaac/' (pose sampling, CUDA rendering, pose update) with direct TensorRT
# inference.
#
# Built by 'colcon build' like any package. Needs a CUDA compiler and the 'tensorrt-cu12' wheel in the Python
# environment of the build (the RoboKudo venv); without them only the Python part is installed and the build warns.
# CV-CUDA and the TensorRT headers (not part of the wheel) are downloaded and verified at configure time.
cmake_minimum_required(VERSION 3.24)
project(robokudo_foundation_pose_tensorrt LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE)
  # colcon passes no build type, speed matters here
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# the venv of the build, it provides TensorRT and decides the Python ABI of the extension
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# the sources are not in a dir named like the module, so a source dir on 'sys.path' cannot shadow the installed
# module, which holds the compiled extension
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR python)
ament_get_python_install_dir(PYTHON_INSTALL_DIR)

# --- prerequisites the system cannot provide via rosdep -----------------------------------------------------------
include(CheckLanguage)
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
  # nvcc < 12.4 (Ubuntu 24.04: 12.0) rejects gcc 13, the default compiler
  set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
endif()
check_language(CUDA)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import os, tensorrt_libs; print(os.path.dirname(tensorrt_libs.__file__))"
  OUTPUT_VARIABLE TENSORRT_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

if(NOT CMAKE_CUDA_COMPILER OR NOT TENSORRT_LIB_DIR)
  message(WARNING
    "robokudo_foundation_pose_tensorrt: building only the Python part. The extension needs a CUDA compiler "
    "(nvidia-cuda-toolkit, found: '${CMAKE_CUDA_COMPILER}') and TensorRT in '${Python3_EXECUTABLE}' "
    "(found: '${TENSORRT_LIB_DIR}', install with 'python3 -m pip install \"tensorrt-cu12>=10.13.3,<11\"').")
  ament_package()
  return()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# static CUDA runtime, so the module does not clash with the libcudart of PyTorch in the same process
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(assimp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# --- downloaded dependencies: pybind11, CV-CUDA and the TensorRT headers --------------------------------------------
include(FetchContent)
# pybind11 >= 2.12 supports NumPy 2, which the CRAM environment uses; Ubuntu 24.04 ships 2.11, whose extensions
# crash on NumPy 2 arrays
FetchContent_Declare(pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(PYBIND11_FINDPYTHON ON)
FetchContent_MakeAvailable(pybind11)

set(CVCUDA_VERSION 0.19.0)
set(CVCUDA_URL https://github.com/CVCUDA/CV-CUDA/releases/download/v${CVCUDA_VERSION})
FetchContent_Declare(cvcuda_lib
  URL ${CVCUDA_URL}/cvcuda-lib-${CVCUDA_VERSION}-cuda12-x86_64-linux.tar.xz
  URL_HASH SHA256=939d80e14c1ed48e5634524aa01bae398df1937b7270317abb7a55c9cb7f3146
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(cvcuda_dev
  URL ${CVCUDA_URL}/cvcuda-dev-${CVCUDA_VERSION}-cuda12-x86_64-linux.tar.xz
  URL_HASH SHA256=abb71db72a908930a92ab6c60936621365380d3c255f84a53bd2224335d9784b
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Populate(cvcuda_lib)
FetchContent_Populate(cvcuda_dev)
# the archives contain 'opt/nvidia/cvcuda0/...', FetchContent strips the single top level directory
set(CVCUDA_LIB_DIR ${cvcuda_lib_SOURCE_DIR}/nvidia/cvcuda0/lib)
set(CVCUDA_INCLUDE_DIR ${cvcuda_dev_SOURCE_DIR}/nvidia/cvcuda0/include)

# the include chain of 'NvInferRuntime.h' of TensorRT 10.13.3 (Apache 2.0, github.com/NVIDIA/TensorRT)
set(TENSORRT_HEADER_URL https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.13.3/include)
set(TENSORRT_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/tensorrt_include)
set(TENSORRT_HEADERS
  "NvInferRuntime.h=c6ef78bbd8d12c8affa22961d136951e87e7b4c2a405fe44aef40b35264e3686"
  "NvInferImpl.h=73cbf708a2f8a57967cb86e97fc38eed096df11c47b6d76090cbd8fcc21993ea"
  "NvInferRuntimeCommon.h=9461c9d07403c5dd1f8a6dea17fa5c3fe5cef7c4290c6768370a1d8139d4ac5d"
  "NvInferRuntimeBase.h=cf979038f28817ebee3b2899322aadda4566dafcd8a060c18bf2bdff511dfc8a"
  "NvInferRuntimePlugin.h=d337b34b7b7efc1c65ba3405a6551389063669cb0c51704f91578b5f817999fc"
  "NvInferPluginBase.h=774e457b2e80f27502ea1aed8132949f6e0788251b5374948e1bfdc9de9fb0ee"
  "NvInferLegacyDims.h=3e10fe5b85545f2e956e6935271badf0b1c72553b698000a61d46d69c5d53030"
  "NvInferVersion.h=b80217d27f09a25be0fb753a7aa832834f2def9cd91d2a5562d278b76c5388c2")
foreach(entry IN LISTS TENSORRT_HEADERS)
  string(REPLACE "=" ";" entry "${entry}")
  list(GET entry 0 header)
  list(GET entry 1 sha256)
  file(DOWNLOAD ${TENSORRT_HEADER_URL}/${header} ${TENSORRT_INCLUDE_DIR}/${header}
       EXPECTED_HASH SHA256=${sha256} TLS_VERIFY ON)
endforeach()

find_library(NVINFER_LIB NAMES libnvinfer.so.10 PATHS ${TENSORRT_LIB_DIR} REQUIRED NO_DEFAULT_PATH)
find_library(CVCUDA_LIB NAMES libcvcuda.so.0 PATHS ${CVCUDA_LIB_DIR} REQUIRED NO_DEFAULT_PATH)
find_library(NVCV_TYPES_LIB NAMES libnvcv_types.so.0 PATHS ${CVCUDA_LIB_DIR} REQUIRED NO_DEFAULT_PATH)

# --- Isaac ROS FoundationPose core (isaac/, see its LICENSE) ---------------------------------------------------------
set(ISAAC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/isaac")
set(IMPL_DIR "${ISAAC_DIR}/src/foundationpose_impl")
set(NVDR_DIR "${ISAAC_DIR}/src/nvdiffrast")

add_library(foundationpose_core STATIC
  ${IMPL_DIR}/mesh_loader.cpp
  ${IMPL_DIR}/pose_sampler.cpp
  ${IMPL_DIR}/pose_renderer.cpp
  ${IMPL_DIR}/pose_transformer.cpp
  ${IMPL_DIR}/foundationpose_sampling.cu
  ${IMPL_DIR}/foundationpose_render.cu
  ${NVDR_DIR}/common/interpolate.cu
  ${NVDR_DIR}/common/rasterize.cu
  ${NVDR_DIR}/common/texture.cu
  ${NVDR_DIR}/common/cudaraster/impl/RasterImpl.cu
  ${NVDR_DIR}/common/common.cpp
  ${NVDR_DIR}/common/cudaraster/impl/Buffer.cpp
  ${NVDR_DIR}/common/cudaraster/impl/CudaRaster.cpp
  ${NVDR_DIR}/common/cudaraster/impl/RasterImpl.cpp
)
set_target_properties(foundationpose_core PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(foundationpose_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-w;--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:-w>)
target_include_directories(foundationpose_core PUBLIC
  ${ISAAC_DIR}/include
  ${ISAAC_DIR}/include/isaac_ros_foundationpose/foundationpose_impl
  ${IMPL_DIR}
  ${ISAAC_DIR}/src
  ${NVDR_DIR}/common
  ${NVDR_DIR}/common/cudaraster
  ${NVDR_DIR}/common/cudaraster/impl
  ${CVCUDA_INCLUDE_DIR}
)
target_link_libraries(foundationpose_core PUBLIC
  CUDA::cudart_static
  Eigen3::Eigen
  assimp::assimp
  ${CVCUDA_LIB}
  ${NVCV_TYPES_LIB}
  ${OpenCV_LIBS}
)

# --- Python extension 'robokudo_foundation_pose_tensorrt._bindings' -------------------------------------------------------
pybind11_add_module(_bindings
  src/trt_engine.cpp
  src/estimator.cpp
  src/bindings.cpp
)
target_include_directories(_bindings PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(_bindings SYSTEM PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(_bindings PRIVATE foundationpose_core ${NVINFER_LIB})
# CV-CUDA is installed next to the extension; libnvinfer is loaded from the running venv by '__init__.py'
set_target_properties(_bindings PROPERTIES
  BUILD_RPATH "${TENSORRT_LIB_DIR};${CVCUDA_LIB_DIR}"
  INSTALL_RPATH "$ORIGIN/libs")

install(TARGETS _bindings LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
file(GLOB CVCUDA_RUNTIME_LIBS "${CVCUDA_LIB_DIR}/libcvcuda.so.*" "${CVCUDA_LIB_DIR}/libnvcv_types.so.*")
install(FILES ${CVCUDA_RUNTIME_LIBS} DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME}/libs)

# --- TensorRT engines: built once per TensorRT version and GPU into the user cache, a no-op afterwards ---------------
option(BUILD_ENGINES "Build the FP16 FoundationPose engines during the build if missing" ON)
if(BUILD_ENGINES)
  add_custom_target(engines ALL
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/python/engines.py --precision fp16 --no-fail
    COMMENT "Checking the FoundationPose TensorRT engines (building them the first time takes a few minutes)"
    VERBATIM)
endif()

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_foundation_pose_tensorrt test/test_foundation_pose_tensorrt.py TIMEOUT 600)
endif()

ament_package()
