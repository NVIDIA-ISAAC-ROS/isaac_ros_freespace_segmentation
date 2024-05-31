# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Dict

from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.descriptions import ComposableNode


class IsaacROSBi3DFreespaceLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

        base_link_frame = LaunchConfiguration('base_link_frame')
        grid_height = LaunchConfiguration('grid_height')
        grid_width = LaunchConfiguration('grid_width')
        grid_resolution = LaunchConfiguration('grid_resolution')

        publish_default_tf = LaunchConfiguration('publish_default_tf')

        return {
            'freespace_segmentation_node': ComposableNode(
                package='isaac_ros_bi3d_freespace',
                plugin='nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode',
                name='freespace_segmentation_node',
                namespace='',
                parameters=[{
                    'base_link_frame': base_link_frame,
                    'camera_frame': interface_specs['camera_frame'],
                    'f_x': interface_specs['focal_length']['f_x'],
                    'f_y': interface_specs['focal_length']['f_y'],
                    'grid_height': grid_height,
                    'grid_width': grid_width,
                    'grid_resolution': grid_resolution,
                    'use_sim_time': True
                }],
                remappings=[
                    ('bi3d_mask', 'bi3d_node/bi3d_output')
                ]
            ),
            'tf_publisher': ComposableNode(
                name='static_transform_publisher',
                package='tf2_ros',
                plugin='tf2_ros::StaticTransformBroadcasterNode',
                parameters=[{
                   'frame_id': base_link_frame,
                   'child_frame_id': interface_specs['camera_frame'],
                   'translation.x': 0.0,
                   'translation.y': 0.0,
                   'translation.z': 0.3,
                   'rotation.x': -0.5,
                   'rotation.y': 0.5,
                   'rotation.z': -0.5,
                   'rotation.w': 0.5
                }],
                condition=IfCondition(publish_default_tf)
            )
        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        return {
            'base_link_frame': DeclareLaunchArgument(
                'base_link_frame',
                default_value='base_link',
                description='The name of the tf2 frame corresponding to the origin of the robot '
                            'base'
            ),
            'grid_height': DeclareLaunchArgument(
                'grid_height',
                default_value='2000',
                description='The desired height of the occupancy grid, in cells'),
            'grid_width': DeclareLaunchArgument(
                'grid_width',
                default_value='2000',
                description='The desired width of the occupancy grid, in cells'),
            'grid_resolution': DeclareLaunchArgument(
                'grid_resolution',
                default_value='0.01',
                description='The desired resolution of the occupancy grid, in m/cell'),
            'publish_default_tf': DeclareLaunchArgument(
                'publish_default_tf',
                default_value='false',
                description='Whether to publish a default tf from the base_link_frame to the '
                            'camera_frame. The default transform locates the camera 0.3m above '
                            'the base_link_frame, with the optical frame Z-axis pointing straight '
                            'forward.'
            )
        }
