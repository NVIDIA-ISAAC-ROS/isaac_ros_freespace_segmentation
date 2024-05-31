// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "occupancy_grid_projector.cu.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace freespace_segmentation
{

__global__ void cuda_process_segmentation_mask(
  const float * segmentation_mask, int8_t * occupancy_grid, int mask_height, int mask_width,
  int grid_height, int grid_width, float grid_resolution, float f_x_, float f_y_,
  float * rotation_matrix, float * translation)
{
  const uint32_t mask_u = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t mask_v = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t mask_index = mask_v * mask_width + mask_u;

  // Ensure segmentation mask coordinates are in-bounds
  if (mask_u >= mask_width || mask_v >= mask_height) {
    return;
  }

  // Ensure pixel corresponds to ground (label of 0)
  if (segmentation_mask[mask_index] != 0.0) {
    return;
  }

  /**
   *  p0: Camera center
   *  p1: Center of pixel on image plane
   *  p2: Intersection of ray from p0 through p1 with ground plane
   */

  // Convert from pixel coordinates (u, v) in pixels to (x, y, z) in distance units
  // Offset to ensure that point passes through center of the pixel
  // f_x_ = s_x_ * f and f_y_ = s_y_ * f
  // p1_x_cam = (mask_u + 0.5f - mask_width / 2.0f) / s_x_;
  // p1_y_cam = (mask_v + 0.5f - mask_height / 2.0f) / s_y_;
  // p1_z_cam = f; // Point exists on image plane
  // Divide each component by f to get the following formulation

  const float p1_x_cam = (mask_u + 0.5f - mask_width / 2.0f) / f_x_;
  const float p1_y_cam = (mask_v + 0.5f - mask_height / 2.0f) / f_y_;
  const float p1_z_cam = 1.0;

  // Apply transformation from camera frame to global frame
  // Extract coefficients from column-major rotation matrix
  const float p1_x_global =
    rotation_matrix[0] * p1_x_cam +
    rotation_matrix[3] * p1_y_cam +
    rotation_matrix[6] * p1_z_cam +
    translation[0];

  const float p1_y_global =
    rotation_matrix[1] * p1_x_cam +
    rotation_matrix[4] * p1_y_cam +
    rotation_matrix[7] * p1_z_cam +
    translation[1];

  const float p1_z_global =
    rotation_matrix[2] * p1_x_cam +
    rotation_matrix[5] * p1_y_cam +
    rotation_matrix[8] * p1_z_cam +
    translation[2];

  // Camera center given by translation
  const float p0_x_global = translation[0];
  const float p0_y_global = translation[1];
  const float p0_z_global = translation[2];

  // Using parametric representation of ray from camera center p0 through pixel p1,
  // find value of parameter t that hits the ground plane to produce p2 with Z=0
  const float t = p0_z_global / (p0_z_global - p1_z_global);

  // Ensure that we only project rays in the 'forwards' direction, out of the camera
  if (t <= 0) {
    return;
  }

  // Calculate p2_global where ray intersects ground plane
  const float p2_x_global = p0_x_global + (p1_x_global - p0_x_global) * t;
  const float p2_y_global = p0_y_global + (p1_y_global - p0_y_global) * t;
  // p2_z_global = 0

  // Calculate grid indices, transforming from global frame to occupancy grid frame
  const uint32_t grid_x = static_cast<uint32_t>(p2_x_global / grid_resolution + grid_width / 2.0f);
  const uint32_t grid_y =
    static_cast<uint32_t>(p2_y_global / grid_resolution + grid_height / 2.0f);
  const uint32_t grid_index = grid_y * grid_width + grid_x;

  // Ensure grid index is within range
  if (grid_x > grid_width || grid_y > grid_height) {
    return;
  }

  // Indicate that the corresponding cell is certainly free (value of 0)
  occupancy_grid[grid_index] = 0;
}

#define CHECK_CUDA_ERRORS(result){checkCudaErrors(result, __FILE__, __LINE__); \
}
inline void checkCudaErrors(cudaError_t result, const char * filename, int line_number)
{
  if (result != cudaSuccess) {
    GXF_LOG_ERROR(
      ("CUDA Error: " + std::string(cudaGetErrorString(result)) +
      " (error code: " + std::to_string(result) + ") at " +
      std::string(filename) + " in line " + std::to_string(line_number)).c_str());
  }
}

uint16_t ceil_div(uint16_t numerator, uint16_t denominator)
{
  uint32_t accumulator = numerator + denominator - 1;
  return accumulator / denominator;
}

void process_segmentation_mask(
  const float * segmentation_mask, int8_t * occupancy_grid, int mask_height, int mask_width,
  int grid_height, int grid_width, float grid_resolution, float f_x,
  float f_y, float * rotation_matrix, float * translation)
{
  // Initialize occupancy grid as all -1 (unknown occupancy status)
  CHECK_CUDA_ERRORS(cudaMemset(occupancy_grid, -1, grid_width * grid_height));

  dim3 block(16, 16);
  dim3 grid(ceil_div(mask_width, 16), ceil_div(mask_height, 16), 1);
  cuda_process_segmentation_mask << < grid, block >> >
  (segmentation_mask, occupancy_grid, mask_height, mask_width, grid_height, grid_width,
  grid_resolution, f_x, f_y, rotation_matrix, translation);

  // Wait for CUDA to finish
  CHECK_CUDA_ERRORS(cudaDeviceSynchronize());
}

}  // namespace freespace_segmentation
}  // namespace isaac_ros
}  // namespace nvidia
