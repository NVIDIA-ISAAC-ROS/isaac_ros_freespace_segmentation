# Isaac ROS Freespace Segmentation

NVIDIA-accelerated, deep-learned freespace segmentation

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_real_opt.gif/"><img alt="Isaac ROS Freespace Segmentation Sample Output" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_real_opt.gif/" width="500px"/></a></div>

## Overview

Isaac ROS Freespace Segmentation contains an ROS 2 package to produce
occupancy grids for navigation. By processing a freespace segmentation
mask with the pose of the robot relative to the ground, Bi3D Freespace
produces an occupancy grid for
[Nav2](https://github.com/ros-planning/navigation2), which is used to
avoid obstacles during navigation. This package is GPU accelerated to
provide real-time, low latency results in a robotics application. Bi3D
Freespace provides an additional occupancy grid source for mobile robots
(ground based).

<div align="center"><a class="reference internal image-reference" href="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_freespace_segmentation_nodegraph.png/"><img alt="Isaac ROS Freespace Segmentation Sample Output" src="https://media.githubusercontent.com/media/NVIDIA-ISAAC-ROS/.github/main/resources/isaac_ros_docs/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_freespace_segmentation_nodegraph.png/" width="700px"/></a></div>

`isaac_ros_bi3d` is used in a graph of nodes to provide a freespace
segmentation mask as one output from a time-synchronized input left and
right stereo image pair. The freespace mask is used by
`isaac_ros_bi3d_freespace` with TF pose of the camera relative to the
ground to compute planar freespace into an occupancy grid as input to
[Nav2](https://github.com/ros-planning/navigation2).

There are multiple methods to predict the occupancy grid as an input to
navigation. None of these methods are perfect; each has limitations on
the accuracy of its estimate from the sensor providing measured
observations. Each sensor has a unique field of view, range to provide
its measured view of the world, and corresponding areas it does not
measure. Bi3D Freespace provides a diverse approach to
identifying obstacles from freespace. Stereo camera input used for this
function is diverse relative to lidar, and has a better vertical field
of view than most lidar units, allowing for perception of low lying
obstacles that lidar can miss. Bi3D Freespace provides a
robust, vision-based complement to lidar occupancy scanning.

## Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

| Sample Graph<br/><br/>                                                                                                                                                                                 | Input Size<br/><br/>     | AGX Orin<br/><br/>                                                                                                                                               | Orin NX<br/><br/>                                                                                                                                               | Orin Nano Super 8GB<br/><br/>                                                                                                                                     | x86_64 w/ RTX 4090<br/><br/>                                                                                                                                      |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Freespace Segmentation Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_bi3d_freespace_benchmark/scripts/isaac_ros_bi3d_fs_node.py)<br/><br/><br/><br/>   | 576p<br/><br/><br/><br/> | [3340 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-agx_orin.json)<br/><br/><br/>1.7 ms @ 30Hz<br/><br/> | [2530 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-orin_nx.json)<br/><br/><br/>1.5 ms @ 30Hz<br/><br/> | [2140 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-orin_nano.json)<br/><br/><br/>1.9 ms @ 30Hz<br/><br/> | [3500 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-x86-4090.json)<br/><br/><br/>0.44 ms @ 30Hz<br/><br/> |
| [Freespace Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/benchmarks/isaac_ros_bi3d_freespace_benchmark/scripts/isaac_ros_bi3d_fs_graph.py)<br/><br/><br/><br/> | 576p<br/><br/><br/><br/> | [40.3 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-agx_orin.json)<br/><br/><br/>79 ms @ 30Hz<br/><br/> | [27.6 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-orin_nx.json)<br/><br/><br/>98 ms @ 30Hz<br/><br/> | [31.8 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-orin_nano.json)<br/><br/><br/>55 ms @ 30Hz<br/><br/> | [102 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-x86-4090.json)<br/><br/><br/>30 ms @ 30Hz<br/><br/>   |

---

## Documentation

Please visit the [Isaac ROS Documentation](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_freespace_segmentation/index.html) to learn how to use this repository.

---

## Packages

* [`isaac_ros_bi3d_freespace`](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_freespace/index.html)
  * [Quickstart](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_freespace/index.html#quickstart)
  * [Try More Examples](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_freespace/index.html#try-more-examples)
  * [API](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_freespace_segmentation/isaac_ros_bi3d_freespace/index.html#api)

## Latest

Update 2024-12-10: Update to be compatible with JetPack 6.1
