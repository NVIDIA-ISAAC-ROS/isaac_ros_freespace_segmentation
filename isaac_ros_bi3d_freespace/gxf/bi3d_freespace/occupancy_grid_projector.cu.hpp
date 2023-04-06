// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_

#include <cstdint>
#include <limits>

#include "cuda.h"
#include "cuda_runtime.h"

namespace nvidia
{
namespace isaac_ros
{
namespace freespace_segmentation
{

void process_segmentation_mask(
  const float * segmentation_mask, int8_t * occupancy_grid, int mask_height, int mask_width,
  int grid_height, int grid_width, float grid_resolution, float f_x,
  float f_y, float * rotation_matrix, float * translation);

}  // namespace freespace_segmentation
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_CU_HPP_
