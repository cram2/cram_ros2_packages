// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_foundationpose/foundationpose_impl/mesh_loader.hpp"

#include <cuda_runtime.h>

#include <atomic>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "isaac_ros_common/cuda_stream.hpp"

#include "assimp/Importer.hpp"  // NOLINT
#include "assimp/postprocess.h"  // NOLINT
#include "assimp/scene.h"  // NOLINT
#include "opencv2/opencv.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

namespace
{
constexpr int kFixTextureMapColor = 128;
}  // namespace

void MeshData::freeMeshDeviceMemory()
{
  if (mesh_vertices_device) {cudaFree(mesh_vertices_device);}
  if (mesh_normals_device) {cudaFree(mesh_normals_device);}
  if (mesh_faces_device) {cudaFree(mesh_faces_device);}
  if (texcoords_device) {cudaFree(texcoords_device);}
  mesh_vertices_device = nullptr;
  mesh_normals_device = nullptr;
  mesh_faces_device = nullptr;
  texcoords_device = nullptr;
}

void MeshData::freeTextureDeviceMemory()
{
  if (texture_map_device) {cudaFree(texture_map_device);}
  texture_map_device = nullptr;
}

MeshData::~MeshData()
{
  freeMeshDeviceMemory();
  freeTextureDeviceMemory();
}

MeshLoader::MeshLoader(cudaStream_t stream)
: stream_(stream), mesh_data_(std::make_shared<MeshData>())
{
}

MeshLoader::~MeshLoader()
{
  mesh_data_.reset();
}

void MeshLoader::load(const std::string & mesh_file_path)
{
  loadMeshData(mesh_file_path);
}

void MeshLoader::tryReload(const std::string & mesh_file_path)
{
  if (mesh_data_->mesh_file_path != mesh_file_path) {
    loadMeshData(mesh_file_path);
  }
}

std::pair<Eigen::Vector3f, Eigen::Vector3f>
MeshLoader::findMinMaxVertex(const aiMesh * mesh)
{
  Eigen::Vector3f min_v{0, 0, 0};
  Eigen::Vector3f max_v{0, 0, 0};
  if (mesh->mNumVertices == 0) {return {min_v, max_v};}

  min_v << mesh->mVertices[0].x, mesh->mVertices[0].y, mesh->mVertices[0].z;
  max_v = min_v;

  for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
    float vx = mesh->mVertices[v].x;
    float vy = mesh->mVertices[v].y;
    float vz = mesh->mVertices[v].z;
    min_v[0] = std::min(min_v[0], vx);
    min_v[1] = std::min(min_v[1], vy);
    min_v[2] = std::min(min_v[2], vz);
    max_v[0] = std::max(max_v[0], vx);
    max_v[1] = std::max(max_v[1], vy);
    max_v[2] = std::max(max_v[2], vz);
  }
  return {min_v, max_v};
}

float MeshLoader::calcMeshDiameter(const aiMesh * mesh)
{
  // The exact diameter is O(V^2) and takes seconds for dense scanned meshes, which stalls every mesh switch.
  // Like the reference FoundationPose implementation, use an evenly strided vertex subset instead.
  constexpr unsigned int kMaxDiameterSamples = 5000;
  const unsigned int stride = std::max(1u, mesh->mNumVertices / kMaxDiameterSamples);

  float max_dist = 0.0f;
  for (unsigned int i = 0; i < mesh->mNumVertices; i += stride) {
    for (unsigned int j = i + stride; j < mesh->mNumVertices; j += stride) {
      aiVector3D diff = mesh->mVertices[i] - mesh->mVertices[j];
      max_dist = std::max(max_dist, diff.Length());
    }
  }
  return max_dist;
}

void MeshLoader::loadTextureData(const std::string & texture_file_path)
{
  cv::Mat rgb_texture_map;

  if (!std::filesystem::exists(texture_file_path)) {
    if (mesh_data_->texture_path.empty() &&
      mesh_data_->texture_map_device != nullptr &&
      mesh_data_->texture_map_width == mesh_data_->num_vertices)
    {
      return;
    }
    rgb_texture_map = cv::Mat(
      1, mesh_data_->num_vertices, CV_8UC3,
      cv::Scalar(kFixTextureMapColor, kFixTextureMapColor, kFixTextureMapColor));
    mesh_data_->has_tex = false;
  } else {
    mesh_data_->texture_path = texture_file_path;
    cv::Mat texture_map = cv::imread(texture_file_path);
    cv::cvtColor(texture_map, rgb_texture_map, cv::COLOR_BGR2RGB);
  }

  if (!rgb_texture_map.isContinuous()) {
    throw std::runtime_error("[MeshLoader] Texture map is not continuous");
  }

  if (mesh_data_->texture_map_device != nullptr) {
    mesh_data_->freeTextureDeviceMemory();
  }

  mesh_data_->texture_map_height = rgb_texture_map.rows;
  mesh_data_->texture_map_width = rgb_texture_map.cols;
  mesh_data_->texture_map_channels = rgb_texture_map.channels();
  static std::atomic<uint64_t> next_texture_id{1};
  mesh_data_->texture_id = next_texture_id++;

  CHECK_CUDA_ERROR(
    cudaMalloc(&mesh_data_->texture_map_device,
    rgb_texture_map.total() * rgb_texture_map.elemSize()),
    "cudaMalloc texture");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(mesh_data_->texture_map_device, rgb_texture_map.data,
    rgb_texture_map.total() * rgb_texture_map.elemSize(),
    cudaMemcpyHostToDevice, stream_),
    "cudaMemcpy texture");
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync texture upload");
}

void MeshLoader::loadMeshData(const std::string & mesh_file_path)
{
  if (!std::filesystem::exists(mesh_file_path)) {
    throw std::runtime_error("[MeshLoader] Mesh file does not exist: " + mesh_file_path);
  }

  if (mesh_data_->mesh_vertices_device != nullptr) {
    mesh_data_->freeMeshDeviceMemory();
  }
  mesh_data_->mesh_file_path = mesh_file_path;

  Assimp::Importer importer;
  const aiScene * scene = importer.ReadFile(
    mesh_file_path,
    aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_SortByPType |
    aiProcess_GenSmoothNormals);

  if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
    throw std::runtime_error(
            std::string("[MeshLoader] Assimp error: ") + importer.GetErrorString());
  }
  if (scene->mNumMeshes == 0) {
    throw std::runtime_error("[MeshLoader] No mesh found in file");
  }

  const aiMesh * mesh = scene->mMeshes[0];
  auto [min_v, max_v] = findMinMaxVertex(mesh);
  mesh_data_->mesh_model_center = (max_v + min_v) / 2.0f;
  mesh_data_->min_vertex = min_v;
  mesh_data_->max_vertex = max_v;
  mesh_data_->mesh_diameter = calcMeshDiameter(mesh);

  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> texcoords;
  std::vector<int32_t> faces;

  for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
    vertices.push_back(mesh->mVertices[v].x - mesh_data_->mesh_model_center[0]);
    vertices.push_back(mesh->mVertices[v].y - mesh_data_->mesh_model_center[1]);
    vertices.push_back(mesh->mVertices[v].z - mesh_data_->mesh_model_center[2]);
    Eigen::Vector3f normal{0.0f, 0.0f, 1.0f};
    if (mesh->HasNormals()) {
      normal << mesh->mNormals[v].x, mesh->mNormals[v].y, mesh->mNormals[v].z;
      if (normal.norm() > 1e-6f) {
        normal.normalize();
      } else {
        normal << 0.0f, 0.0f, 1.0f;
      }
    }
    normals.push_back(normal[0]);
    normals.push_back(normal[1]);
    normals.push_back(normal[2]);
    if (mesh->mTextureCoords[0]) {
      texcoords.push_back(mesh->mTextureCoords[0][v].x);
      texcoords.push_back(1.0f - mesh->mTextureCoords[0][v].y);
    }
  }

  for (unsigned int f = 0; f < mesh->mNumFaces; f++) {
    const aiFace & face = mesh->mFaces[f];
    if (face.mNumIndices != 3) {
      throw std::runtime_error("[MeshLoader] Non-triangle face found");
    }
    for (unsigned int i = 0; i < face.mNumIndices; i++) {
      faces.push_back(face.mIndices[i]);
    }
  }

  mesh_data_->num_vertices = static_cast<int>(vertices.size() / 3);
  mesh_data_->num_texcoords = static_cast<int>(texcoords.size() / 2);
  mesh_data_->num_faces = static_cast<int>(faces.size() / 3);

  if (mesh_data_->num_vertices == 0 || mesh_data_->num_faces == 0) {
    throw std::runtime_error("[MeshLoader] Empty mesh data");
  }

  mesh_data_->mesh_vertices = Eigen::Map<
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
    vertices.data(), mesh_data_->num_vertices, 3);

  size_t verts_bytes = vertices.size() * sizeof(float);
  size_t normals_bytes = normals.size() * sizeof(float);
  size_t faces_bytes = faces.size() * sizeof(int32_t);
  size_t tex_bytes = texcoords.size() * sizeof(float);

  CHECK_CUDA_ERROR(cudaMalloc(&mesh_data_->mesh_vertices_device, verts_bytes), "malloc vertices");
  CHECK_CUDA_ERROR(cudaMalloc(&mesh_data_->mesh_normals_device, normals_bytes), "malloc normals");
  CHECK_CUDA_ERROR(cudaMalloc(&mesh_data_->mesh_faces_device, faces_bytes), "malloc faces");
  if (tex_bytes > 0) {
    CHECK_CUDA_ERROR(cudaMalloc(&mesh_data_->texcoords_device, tex_bytes), "malloc texcoords");
  }

  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      mesh_data_->mesh_vertices_device, vertices.data(),
      verts_bytes, cudaMemcpyHostToDevice, stream_), "memcpy vertices");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      mesh_data_->mesh_normals_device, normals.data(),
      normals_bytes, cudaMemcpyHostToDevice, stream_), "memcpy normals");
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(
      mesh_data_->mesh_faces_device, faces.data(),
      faces_bytes, cudaMemcpyHostToDevice, stream_), "memcpy faces");
  if (mesh_data_->texcoords_device) {
    CHECK_CUDA_ERROR(
      cudaMemcpyAsync(
        mesh_data_->texcoords_device, texcoords.data(),
        tex_bytes, cudaMemcpyHostToDevice, stream_), "memcpy texcoords");
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_), "sync mesh upload");

  std::string texture_path_str;
  if (mesh->mMaterialIndex >= 0) {
    aiMaterial * material = scene->mMaterials[mesh->mMaterialIndex];
    if (material) {
      aiString tex_path;
      if (material->GetTexture(aiTextureType_DIFFUSE, 0, &tex_path) == AI_SUCCESS) {
        std::filesystem::path mesh_dir = std::filesystem::path(mesh_file_path).parent_path();
        texture_path_str = (mesh_dir / tex_path.C_Str()).string();
      }
    }
  }

  if (texture_path_str != mesh_data_->texture_path ||
    mesh_data_->texture_map_device == nullptr || !mesh_data_->has_tex)
  {
    loadTextureData(texture_path_str);
  }
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia
