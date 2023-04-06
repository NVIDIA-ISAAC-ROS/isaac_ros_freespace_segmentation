# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gxf_occupancy_grid_projector LANGUAGES C CXX CUDA)

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GXF ${ISAAC_ROS_GXF_VERSION} MODULE REQUIRED
    COMPONENTS
    std
    cuda
    multimedia
)
include(YamlCpp)

# Occupancy Grid Projector extension
add_library(gxf_occupancy_grid_projector SHARED
    occupancy_grid_projector_ext.cpp
    occupancy_grid_projector.cpp
    occupancy_grid_projector.cu.cpp
    occupancy_grid_projector.cu.hpp
    occupancy_grid_projector.hpp
)
# Mark as CUDA files with non-standard extensions
set_source_files_properties(
  occupancy_grid_projector.cu.cpp
  occupancy_grid_projector.cu.hpp
  PROPERTIES LANGUAGE CUDA
)
set(CMAKE_INCLUDE_CURRENT_DIR TRUE)
target_link_libraries(gxf_occupancy_grid_projector
  PUBLIC
    CUDA::cudart
    Eigen3::Eigen
    GXF::std
    GXF::cuda
    GXF::multimedia
    yaml-cpp
)