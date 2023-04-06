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
#include "occupancy_grid_projector.hpp"

#include "gxf/std/extension_factory_helper.hpp"

GXF_EXT_FACTORY_BEGIN()

GXF_EXT_FACTORY_SET_INFO(
  0xef1862d43a7911ed, 0xa2610242ac120002,
  "OccupancyGridProjectorExtension",
  "Isaac ROS Occupancy Grid Projector Extension", "NVIDIA", "0.0.1",
  "LICENSE");

GXF_EXT_FACTORY_ADD(
  0x0f0237783a7a11ed, 0xa2610242ac120002,
  nvidia::isaac_ros::freespace_segmentation::OccupancyGridProjector, nvidia::gxf::Codelet,
  "Projects a segmentation mask to the ground plane as an occupancy grid");

GXF_EXT_FACTORY_END()
