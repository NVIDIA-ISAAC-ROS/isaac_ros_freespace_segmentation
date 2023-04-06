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
#ifndef NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_OCCUPANCY_GRID_PROJECTOR_HPP_
#define NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_OCCUPANCY_GRID_PROJECTOR_HPP_

#include <string>

#include <Eigen/Dense>

#include "gxf/std/codelet.hpp"
#include "gxf/std/memory_buffer.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "occupancy_grid_projector.cu.hpp"

namespace nvidia
{
namespace isaac_ros
{
namespace freespace_segmentation
{

class OccupancyGridProjector : public gxf::Codelet
{
public:
  gxf_result_t registerInterface(gxf::Registrar * registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() noexcept override;

private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> mask_receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_transmitter_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  // Parameters
  gxf::Parameter<std::vector<double>> projection_transform_param_;
  gxf::Parameter<std::vector<double>> intrinsics_param_;
  gxf::Parameter<int> grid_height_param_;
  gxf::Parameter<int> grid_width_param_;
  gxf::Parameter<double> grid_resolution_param_;

  // Parsed parameters
  float * rotation_matrix_device_{};
  float * translation_device_{};
  float f_x_{};
  float f_y_{};
  int grid_height_{};
  int grid_width_{};
  float grid_resolution_{};
};

}  // namespace freespace_segmentation
}  // namespace isaac_ros
}  // namespace nvidia

#endif  // NVIDIA_ISAAC_ROS_GXF_EXTENSIONS_OCCUPANCY_GRID_PROJECTOR_HPP_
