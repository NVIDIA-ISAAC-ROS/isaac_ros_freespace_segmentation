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

#include <string>
#include <utility>

#include "occupancy_grid_projector.cu.hpp"
#include "occupancy_grid_projector.hpp"

#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace isaac_ros {
namespace freespace_segmentation {

gxf_result_t OccupancyGridProjector::registerInterface(gxf::Registrar * registrar) {
  gxf::Expected<void> result;

  result &= registrar->parameter(
    mask_receiver_, "mask_receiver", "Mask Receiver",
    "Input for segmentation mask from Bi3D postprocessor");
  result &= registrar->parameter(
    output_transmitter_, "output_transmitter", "Output transmitter",
    "Output containing the projected ground-plane occupancy grid");
  result &= registrar->parameter(
    allocator_, "allocator", "Allocator",
    "Allocator instance for output");

  result &= registrar->parameter(
    projection_transform_param_, "projection_transform",
    "Projection transform",
    "Transform from camera frame to ground plane frame");
  result &= registrar->parameter(
    intrinsics_param_, "intrinsics",
    "Intrinsics",
    "The camera intrinsics");
  result &= registrar->parameter(
    grid_height_param_, "grid_height",
    "Grid height",
    "Number of rows in the occupancy grid");
  result &= registrar->parameter(
    grid_width_param_, "grid_width",
    "Grid width",
    "Number of columns in the occupancy grid");
  result &= registrar->parameter(
    grid_resolution_param_, "grid_resolution",
    "Grid resolution",
    "Occupancy grid resolution in meters per cell");

  return gxf::ToResultCode(result);
}

gxf_result_t OccupancyGridProjector::start() {
  // Extract 3D transform from camera frame to ground frame from parameter
  auto projection = projection_transform_param_.get();
  if (projection.size() != 7) {
    GXF_LOG_ERROR("Expected 3D transform vector to be length 7 but got %lu", projection.size());
    return GXF_FAILURE;
  }
  auto translation =
    Eigen::Vector3f{static_cast<float>(projection.at(0)),
    static_cast<float>(projection.at(1)), static_cast<float>(projection.at(2))};

  // Allocate and copy translation to device
  if (cudaMalloc(&translation_device_, sizeof(float) * translation.size()) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to allocate translation on device");
    return GXF_FAILURE;
  }
  if (cudaMemcpy(
      translation_device_, translation.data(),
      sizeof(float) * translation.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy translation to device");
    return GXF_FAILURE;
  }

  // Parse quaternion (xyzw) as Eigen quaternion (wxyz)
  auto q =
    Eigen::Quaternionf{static_cast<float>(projection.at(6)), static_cast<float>(projection.at(3)),
    static_cast<float>(projection.at(4)), static_cast<float>(projection.at(5))};

  // Convert quaternion to rotation matrix
  auto rotation_matrix = q.toRotationMatrix();

  // Allocate and copy rotation matrix to device
  if (cudaMalloc(&rotation_matrix_device_, sizeof(float) * rotation_matrix.size()) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to allocate rotation matrix on device");
    return GXF_FAILURE;
  }
  if (cudaMemcpy(
      rotation_matrix_device_, rotation_matrix.data(),
      sizeof(float) * rotation_matrix.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy rotation matrix to device");
    return GXF_FAILURE;
  }

  // Extract camera intrinsics from parameter
  auto intrinsics = intrinsics_param_.get();
  if (intrinsics.size() != 2) {
    GXF_LOG_ERROR("Expected intrinsics vector to be length 2 but got %lu", intrinsics.size());
    return GXF_FAILURE;
  }
  f_x_ = static_cast<float>(intrinsics.at(0));
  f_y_ = static_cast<float>(intrinsics.at(1));

  // Extract occupancy grid dimension parameters
  grid_height_ = grid_height_param_.get();
  grid_width_ = grid_width_param_.get();
  grid_resolution_ = static_cast<float>(grid_resolution_param_.get());

  return GXF_SUCCESS;
}

gxf_result_t OccupancyGridProjector::stop() noexcept {
  if (cudaFree(rotation_matrix_device_) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to free rotation matrix");
    return GXF_FAILURE;
  }
  if (cudaFree(translation_device_) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to free rotation matrix");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t OccupancyGridProjector::tick() {
  // Retrieve segmentation mask from Bi3D postprocessor
  const auto maybe_mask_message = mask_receiver_->receive();
  if (!maybe_mask_message) {
    return gxf::ToResultCode(maybe_mask_message);
  }

  const auto maybe_mask = maybe_mask_message.value().get<gxf::VideoBuffer>("frame");
  if (!maybe_mask) {
    return gxf::ToResultCode(maybe_mask);
  }

  auto segmentation_mask = maybe_mask.value();

  // Allocate output message
  auto maybe_occupancy_grid_message = gxf::Entity::New(context());
  if (!maybe_occupancy_grid_message) {
    GXF_LOG_ERROR("Failed to allocate occupancy grid message");
    return gxf::ToResultCode(maybe_occupancy_grid_message);
  }
  auto occupancy_grid_message = maybe_occupancy_grid_message.value();

  // Populate primitive metadata
  auto maybe_resolution = occupancy_grid_message.add<float>("resolution");
  if (!maybe_resolution) {
    GXF_LOG_ERROR("Failed to allocate resolution");
    return gxf::ToResultCode(maybe_resolution);
  }
  *maybe_resolution.value() = grid_resolution_;

  auto maybe_width = occupancy_grid_message.add<int>("width");
  if (!maybe_width) {
    GXF_LOG_ERROR("Failed to allocate width");
    return gxf::ToResultCode(maybe_width);
  }
  *maybe_width.value() = grid_width_;

  auto maybe_height = occupancy_grid_message.add<int>("height");
  if (!maybe_height) {
    GXF_LOG_ERROR("Failed to allocate height");
    return gxf::ToResultCode(maybe_height);
  }
  *maybe_height.value() = grid_height_;

  // Allocate output tensor for the occupancy grid origin
  auto maybe_origin = occupancy_grid_message.add<gxf::Tensor>("origin");
  if (!maybe_origin) {
    GXF_LOG_ERROR("Failed to allocate origin");
    return gxf::ToResultCode(maybe_origin);
  }
  auto origin = maybe_origin.value();

  // Initializing the origin
  auto result = origin->reshape<double>(
    nvidia::gxf::Shape{7},
    nvidia::gxf::MemoryStorageType::kDevice, allocator_);
  if (!result) {
    GXF_LOG_ERROR("Failed to reshape origin to (7,)");
    return gxf::ToResultCode(result);
  }

  // Populate the origin with center of map
  std::array<double, 7> origin_pose{
    -grid_width_ / 2.0 * grid_resolution_,   // translation x
    -grid_height_ / 2.0 * grid_resolution_,  // translation y
    0,                                      // translation z
    0, 0, 0, 1                              // rotation
  };

  if (cudaMemcpy(
      origin->pointer(), origin_pose.data(), sizeof(double) * origin_pose.size(),
      cudaMemcpyHostToDevice) != cudaSuccess) {
    GXF_LOG_ERROR("Failed to copy origin to device");
    return GXF_FAILURE;
  }

  // Allocate output tensor for the occupancy grid data
  auto maybe_occupancy_grid = occupancy_grid_message.add<gxf::Tensor>("data");
  if (!maybe_occupancy_grid) {
    GXF_LOG_ERROR("Failed to allocate data");
    return gxf::ToResultCode(maybe_occupancy_grid);
  }
  auto occupancy_grid = maybe_occupancy_grid.value();

  // Initializing the occupancy grid
  result = occupancy_grid->reshape<int8_t>(
    nvidia::gxf::Shape{grid_height_, grid_width_},
    nvidia::gxf::MemoryStorageType::kDevice, allocator_);
  if (!result) {
    GXF_LOG_ERROR(
      "Failed to reshape occupancy grid to (%d, %d)",
      grid_height_, grid_width_);
    return gxf::ToResultCode(result);
  }

  // Process segmentation map
  process_segmentation_mask(
    reinterpret_cast<float *>(segmentation_mask->pointer()), occupancy_grid->data<int8_t>().value(),
    segmentation_mask->video_frame_info().height,
    segmentation_mask->video_frame_info().width,
    grid_height_, grid_width_, grid_resolution_, f_x_, f_y_, rotation_matrix_device_,
    translation_device_);

  // Add timestamp
  std::string timestamp_name{"timestamp"};
  auto maybe_mask_timestamp = maybe_mask_message->get<nvidia::gxf::Timestamp>();
  if (!maybe_mask_timestamp) {
    GXF_LOG_ERROR("Failed to get a timestamp from Bi3D segmentation mask output");
  }
  auto out_timestamp = occupancy_grid_message.add<gxf::Timestamp>(timestamp_name.c_str());
  if (!out_timestamp) {return GXF_FAILURE;}
  *out_timestamp.value() = *maybe_mask_timestamp.value();

  // Publish message
  return gxf::ToResultCode(
    output_transmitter_->publish(std::move(occupancy_grid_message)));
}

}  // namespace freespace_segmentation
}  // namespace isaac_ros
}  // namespace nvidia
