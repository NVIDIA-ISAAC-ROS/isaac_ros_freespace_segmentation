# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'featnet_engine_file_path',
            default_value='',
            description='The absolute path to the Bi3D Featnet TensorRT engine plan'),
        DeclareLaunchArgument(
            'segnet_engine_file_path',
            default_value='',
            description='The absolute path to the Bi3D Segnet TensorRT engine plan'),
        DeclareLaunchArgument(
            'max_disparity_values',
            default_value='64',
            description='The maximum number of disparity values given for Bi3D inference'),
        DeclareLaunchArgument(
            'base_link_frame',
            default_value='base_link',
            description='The name of the tf2 frame corresponding to the origin of the robot base'),
        DeclareLaunchArgument(
            'camera_frame',
            default_value='left_cam',
            description='The name of the tf2 frame corresponding to the camera center'),
        DeclareLaunchArgument(
            'f_x',
            default_value='478.9057',
            description='The number of pixels per distance unit in the x dimension'),
        DeclareLaunchArgument(
            'f_y',
            default_value='459.7495',
            description='The number of pixels per distance unit in the y dimension'),
        DeclareLaunchArgument(
            'grid_height',
            default_value='2000',
            description='The desired height of the occupancy grid, in cells'),
        DeclareLaunchArgument(
            'grid_width',
            default_value='2000',
            description='The desired width of the occupancy grid, in cells'),
        DeclareLaunchArgument(
            'grid_resolution',
            default_value='0.01',
            description='The desired resolution of the occupancy grid, in m/cell'),
        DeclareLaunchArgument(
            'module_id',
            default_value='2',
            description='Index specifying the stereo camera module to use.'),
    ]

    module_id = LaunchConfiguration('module_id')

    # Bi3DNode parameters
    featnet_engine_file_path = LaunchConfiguration('featnet_engine_file_path')
    segnet_engine_file_path = LaunchConfiguration('segnet_engine_file_path')
    max_disparity_values = LaunchConfiguration('max_disparity_values')

    # FreespaceSegmentationNode parameters
    base_link_frame = LaunchConfiguration('base_link_frame')
    camera_frame = LaunchConfiguration('camera_frame')
    f_x_ = LaunchConfiguration('f_x')
    f_y_ = LaunchConfiguration('f_y')
    grid_height = LaunchConfiguration('grid_height')
    grid_width = LaunchConfiguration('grid_width')
    grid_resolution = LaunchConfiguration('grid_resolution')

    argus_stereo_node = ComposableNode(
        name='argus_stereo',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusStereoNode',
        parameters=[{'module_id': module_id}],
    )

    left_rectify_node = ComposableNode(
        name='left_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', 'left/image_raw'),
            ('camera_info', 'left/camera_info'),
            ('image_rect', 'left/image_rect'),
            ('camera_info_rect', 'left/camera_info')
        ]
    )

    right_rectify_node = ComposableNode(
        name='right_rectify_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::RectifyNode',
        parameters=[{
            'output_width': 960,
            'output_height': 576,
        }],
        remappings=[
            ('image_raw', 'right/image_raw'),
            ('camera_info', 'right/camera_info'),
            ('image_rect', 'right/image_rect'),
            ('camera_info_rect', 'right/camera_info')
        ]
    )

    bi3d_node = ComposableNode(
        name='bi3d_node',
        package='isaac_ros_bi3d',
        plugin='nvidia::isaac_ros::bi3d::Bi3DNode',
        parameters=[{
                'featnet_engine_file_path': featnet_engine_file_path,
                'segnet_engine_file_path': segnet_engine_file_path,
                'max_disparity_values': max_disparity_values,
                }],
        remappings=[('bi3d_node/bi3d_output', 'bi3d_mask'),
                    ('left_image_bi3d', 'left/image_rect'),
                    ('left_camera_info_bi3d', 'right/camera_info'),
                    ('right_image_bi3d', 'right/image_rect'),
                    ('right_camera_info_bi3d', 'right/camera_info')]
    )

    freespace_segmentation_node = ComposableNode(
        name='freespace_segmentation_node',
        package='isaac_ros_bi3d_freespace',
        plugin='nvidia::isaac_ros::bi3d_freespace::FreespaceSegmentationNode',
        parameters=[{
            'base_link_frame': base_link_frame,
            'camera_frame': camera_frame,
            'f_x': f_x_,
            'f_y': f_y_,
            'grid_height': grid_height,
            'grid_width': grid_width,
            'grid_resolution': grid_resolution,
            'use_sim_time': True
        }])

    tf_publisher = ComposableNode(
        name='static_transform_publisher',
        package='tf2_ros',
        plugin='tf2_ros::StaticTransformBroadcasterNode',
        parameters=[{
            'frame_id': base_link_frame,
            'child_frame_id': 'camera',
            'translation.x': 0.0,
            'translation.y': 0.0,
            'translation.z': 0.1,
            'rotation.x': 0.0,
            'rotation.y': 0.0,
            'rotation.z': 0.0,
            'rotation.w': 1.0
        }])

    container = ComposableNodeContainer(
        name='bi3d_freespace_container',
        namespace='bi3d_freespace',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            bi3d_node,
            freespace_segmentation_node,
            left_rectify_node,
            right_rectify_node,
            argus_stereo_node,
            tf_publisher
        ],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info',
                   '--log-level', 'color_format_convert:=info',
                   '--log-level', 'NitrosImage:=info',
                   '--log-level', 'NitrosNode:=info'
                   ],
    )

    final_launch_description = launch_args + [container]
    return (launch.LaunchDescription(final_launch_description))
