# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22.1)
project(isaac_ros_bi3d_freespace LANGUAGES C CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# FreespaceSegmentation_node
ament_auto_add_library(freespace_segmentation_node SHARED src/freespace_segmentation_node.cpp)
rclcpp_components_register_nodes(freespace_segmentation_node "nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode")
set(node_plugins "${node_plugins}nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode;$<TARGET_FILE:freespace_segmentation_node>\n")

### Install extensions built from source

# Bi3D Freespace
add_subdirectory(gxf/bi3d_freespace)
install(TARGETS gxf_occupancy_grid_projector DESTINATION share/${PROJECT_NAME}/gxf/lib/bi3d_freespace)

### End extensions

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()


  # The FindPythonInterp and FindPythonLibs modules are removed
  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  find_package(launch_testing_ament_cmake REQUIRED)
  add_launch_test(test/isaac_ros_freespace_segmentation_pol.py TIMEOUT "1000")
endif()

ament_auto_package(INSTALL_TO_SHARE config launch)
