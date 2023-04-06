# Isaac ROS Freespace Segmentation

<div align="center"><img alt="Isaac ROS Freespace Segmentation Sample Output" src="resources/isaac_ros_bi3d_real_opt.gif" width="500px"/></div>

## Overview

Isaac ROS Freespace Segmentation contains an ROS 2 package to produce occupancy grids for navigation. By processing a freespace segmentation mask with the pose of the robot relative to the ground, Bi3D Freespace produces an occupancy grid for [Nav2](https://github.com/ros-planning/navigation2), which is used to avoid obstacles during navigation. This package is GPU accelerated to provide real-time, low latency results in a robotics application. Bi3D Freespace provides an additional occupancy grid source for mobile robots (ground based).

<div align="center"><img alt="Isaac ROS Freespace Segmentation Sample Output" src="resources/isaac_ros_freespace_segmentation_nodegraph.png" width="700px"/></div>

`isaac_ros_bi3d` is used in a graph of nodes to provide a freespace segmentation mask as one output from a time-synchronized input left and right stereo image pair. The freespace mask is used by `isaac_ros_bi3d_freespace` with TF pose of the camera relative to the ground to compute planar freespace into an occupancy grid as input to [Nav2](https://github.com/ros-planning/navigation2).

There are multiple methods to predict the occupancy grid as an input to navigation. None of these methods are perfect; each has limitations on the accuracy of its estimate from the sensor providing measured observations. Each sensor has a unique field of view, range to provide its measured view of the world, and corresponding areas it does not measure. `isaac_ros_bi3d_freespace` provides a diverse approach to identifying obstacles from freespace. Stereo camera input used for this function is diverse relative to lidar, and has a better vertical field of view than most lidar units, allowing for perception of low lying obstacles that lidar can miss. Isaac_ros_bi3d_freespace provides a robust, vision-based complement to lidar occupancy scanning.

### Isaac ROS NITROS Acceleration

This package is powered by [NVIDIA Isaac Transport for ROS (NITROS)](https://developer.nvidia.com/blog/improve-perception-performance-for-ros-2-applications-with-nvidia-isaac-transport-for-ros/), which leverages type adaptation and negotiation to optimize message formats and dramatically accelerate communication between participating nodes.

## Performance

The following table summarizes the per-platform performance statistics of sample graphs that use this package, with links included to the full benchmark output. These benchmark configurations are taken from the [Isaac ROS Benchmark](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark#list-of-isaac-ros-benchmarks) collection, based on the [`ros2_benchmark`](https://github.com/NVIDIA-ISAAC-ROS/ros2_benchmark) framework.

| Sample Graph                                                                                                                                  | Input Size | AGX Orin                                                                                                                                     | Orin NX                                                                                                                                      | Orin Nano 8GB                                                                                                                                      | x86_64 w/ RTX 3060 Ti                                                                                                                                  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Freespace Segmentation Node](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_bi3d_fs_node.py)   | 576p       | [1680 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-agx_orin.json)<br>1.1 ms | [1240 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-orin_nx.json)<br>1.3 ms  | [926 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-orin_nano_8gb.json)<br>1.7 ms   | [2830 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_node-x86_64_rtx_3060Ti.json)<br>0.31 ms |
| [Freespace Segmentation Graph](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/scripts//isaac_ros_bi3d_fs_graph.py) | 576p       | [53.7 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-agx_orin.json)<br>41 ms | [28.1 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-orin_nx.json)<br>120 ms | [19.4 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-orin_nano_8gb.json)<br>100 ms | [167 fps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_benchmark/blob/main/results/isaac_ros_bi3d_fs_graph-x86_64_rtx_3060Ti.json)<br>26 ms   |


## Table of Contents

- [Isaac ROS Freespace Segmentation](#isaac-ros-freespace-segmentation)
  - [Overview](#overview)
    - [Isaac ROS NITROS Acceleration](#isaac-ros-nitros-acceleration)
  - [Performance](#performance)
  - [Table of Contents](#table-of-contents)
  - [Latest Update](#latest-update)
  - [Supported Platforms](#supported-platforms)
    - [Docker](#docker)
  - [Quickstart](#quickstart)
  - [Next Steps](#next-steps)
    - [Try More Examples](#try-more-examples)
  - [Package Reference](#package-reference)
    - [`isaac_ros_bi3d_freespace`](#isaac_ros_bi3d_freespace)
      - [Usage](#usage)
      - [ROS Parameters](#ros-parameters)
      - [ROS Topics Subscribed](#ros-topics-subscribed)
      - [ROS Topics Published](#ros-topics-published)
  - [Troubleshooting](#troubleshooting)
    - [Isaac ROS Troubleshooting](#isaac-ros-troubleshooting)
    - [DNN and Triton Troubleshooting](#dnn-and-triton-troubleshooting)
  - [Updates](#updates)

## Latest Update

Update 2023-04-05: Initial release

## Supported Platforms

This package is designed and tested to be compatible with ROS 2 Humble running on [Jetson](https://developer.nvidia.com/embedded-computing) or an x86_64 system with an NVIDIA GPU.

> **Note**: Versions of ROS 2 earlier than Humble are **not** supported. This package depends on specific ROS 2 implementation features that were only introduced beginning with the Humble release.

| Platform | Hardware                                                                                                                                                                                                 | Software                                                                                                           | Notes                                                                                                                                                                                       |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Jetson   | [Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) <br> [Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-agx-xavier/) | [JetPack 5.1.1](https://developer.nvidia.com/embedded/jetpack)                                                     | For best performance, ensure that the [power settings](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/SD/PlatformPowerAndPerformance.html) are configured appropriately. |
| x86_64   | NVIDIA GPU                                                                                                                                                                                               | [Ubuntu 20.04+](https://releases.ubuntu.com/20.04/) <br> [CUDA 11.8+](https://developer.nvidia.com/cuda-downloads) |

### Docker

To simplify development, we strongly recommend leveraging the Isaac ROS Dev Docker images by following [these steps](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md). This will streamline your development environment setup with the correct versions of dependencies on both Jetson and x86_64 platforms.

> **Note**: All Isaac ROS quick start guides, tutorials, and examples have been designed with the Isaac ROS Docker images as a prerequisite.

## Quickstart

1. Set up your development environment by following the instructions [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/dev-env-setup.md).

2. Clone this repository and its dependencies under `~/workspaces/isaac_ros-dev/src`.

      ```bash
      cd ~/workspaces/isaac_ros-dev/src && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros && 
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline &&
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation &&
      git clone https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_freespace_segmentation
      ```

3. Pull down a rosbag of sample data:

      ```bash
      cd ~/workspaces/isaac_ros-dev/src/isaac_ros_proximity_segmentation && 
      git lfs pull -X "" -I "resources/rosbags/bi3dnode_rosbag"
      ```

4. Launch the Docker container using the `run_dev.sh` script:

      ```bash
      cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common &&
      ./scripts/run_dev.sh
      ```

5. Download model files for Bi3D (refer to the [Model Preparation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation/blob/main/README.md#model-preparation) section for more information):

      ```bash
      mkdir -p /tmp/models/bi3d &&
      cd /tmp/models/bi3d &&
      wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/featnet.onnx' &&
      wget 'https://api.ngc.nvidia.com/v2/models/nvidia/isaac/bi3d_proximity_segmentation/versions/2.0.0/files/segnet.onnx'
      ```

6. Convert the `.onnx` model files to TensorRT engine plan files (refer to the [Model Preparation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_proximity_segmentation/blob/main/README.md#model-preparation) section for more information):

    If using Jetson (Generate engine plans with DLA support enabled):

      ```bash
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_featnet.plan \
      --onnx=/tmp/models/bi3d/featnet.onnx \
      --int8 --useDLACore=0 --allowGPUFallback &&
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_segnet.plan \
      --onnx=/tmp/models/bi3d/segnet.onnx \
      --int8 --useDLACore=0 --allowGPUFallback
      ```

    If using x86_64:

      ```bash
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_featnet.plan \
      --onnx=/tmp/models/bi3d/featnet.onnx --int8 &&
      /usr/src/tensorrt/bin/trtexec --saveEngine=/tmp/models/bi3d/bi3dnet_segnet.plan \
      --onnx=/tmp/models/bi3d/segnet.onnx --int8
      ```

    > **Note**: The engine plans generated using the x86_64 commands will also work on Jetson, but performance will be reduced.

7. Build and source the workspace:  

      ```bash
      cd /workspaces/isaac_ros-dev &&
      colcon build --symlink-install &&
      source install/setup.bash
      ```

8. (Optional) Run tests to verify complete and correct installation:  

      ```bash
      colcon test --executor sequential
      ```

9. Run the launch file to spin up a demo of this package:

      ```bash
      ros2 launch isaac_ros_bi3d_freespace isaac_ros_bi3d_freespace.launch.py featnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_featnet.plan \
      segnet_engine_file_path:=/tmp/models/bi3d/bi3dnet_segnet.plan \
      max_disparity_values:=10
      ```

10. Open a **second** terminal inside the Docker container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

11. Play the rosbag file to simulate image streams from the cameras:

    ```bash
    ros2 bag play --loop src/isaac_ros_proximity_segmentation/resources/rosbags/bi3dnode_rosbag
    ```

12. Open a **third** terminal inside the Docker container:

    ```bash
    cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && \
      ./scripts/run_dev.sh
    ```

13. Visualize the occupancy grid in RViz.

    Start RViz:

    ```bash
    rviz2
    ```

    In the left pane, change **Fixed Frame** to `base_link`.

    In the left pane, click the **Add** button, then select **By topic** followed by **Map** to add the occupancy grid.

    <div align="center"><img alt="RViz Output" src="resources/Rviz_quickstart.png" width="500px"/></div>

## Next Steps

### Try More Examples

To continue your exploration, check out the following suggested examples:

| Example                                                                                                  | Dependencies                                                                                                         |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| [Tutorial with RealSense, Bi3D, and Freespace Segmentation](./docs/tutorial-bi3d-freespace-realsense.md) | [`realsense-ros`](https://github.com/NVIDIA-ISAAC-ROS/.github/blob/main/profile/realsense-setup.md) |
| [Tutorial for Freespace Segmentation with Isaac Sim](./docs/tutorial-bi3d-freespace-isaac-sim.md)        | --                                                                                                                   |

## Package Reference

### `isaac_ros_bi3d_freespace`

#### Usage

```bash
ros2 launch isaac_ros_bi3d_freespace isaac_ros_freespace_segmentation.launch.py base_link_frame:=<"name of base link"> camera_frame:=<"name of camera frame"> f_x:=<"focal length in pixels in x dimension"> f_y:=<"focal length in pixels in y dimension"> grid_width:=<"desired grid width"> grid_height:=<"desired grid height"> grid_resolution:=<"desired grid resolution">
```

#### ROS Parameters

| ROS Parameter     | Type          | Default     | Description                                                     |
| ----------------- | ------------- | ----------- | --------------------------------------------------------------- |
| `base_link_frame` | `std::string` | `base_link` | The name of the `tf2` frame attached to the robot base          |
| `camera_frame`    | `std::string` | `camera`    | The name of the `tf2` frame attached to the camera              |
| `f_x`             | `double`      | `0.0`       | The focal length in pixels in x dimension                       |
| `f_y`             | `double`      | `0.0`       | The focal length in pixels in y dimension                       |
| `grid_width`      | `int`         | `100`       | The width of the output occupancy grid, in number of cells      |
| `grid_height`     | `int`         | `100`       | The height of the output occupancy grid, in number of cells     |
| `grid_resolution` | `double`      | `0.01`      | The resolution of the output occupancy grid, in meters per cell |

#### ROS Topics Subscribed

| ROS Topic                        | Interface                                                                                                              | Description                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `freespace_segmentation/mask_in` | [stereo_msgs/DisparityImage](https://github.com/ros2/common_interfaces/blob/humble/stereo_msgs/msg/DisparityImage.msg) | The input disparity image, with pixels corresponding to ground labelled as 0 |

> **Limitation**: For all input images, both the height and width must be an even number of pixels.

#### ROS Topics Published

| ROS Topic                               | Interface                                                                                                      | Description                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| `freespace_segmentation/occupancy_grid` | [nav_msgs/OccupancyGrid](https://github.com/ros2/common_interfaces/blob/humble/nav_msgs/msg/OccupancyGrid.msg) | The output occupancy grid, with cells marked as 0 if free |

## Troubleshooting

### Isaac ROS Troubleshooting

Check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common/blob/main/docs/troubleshooting.md) for solutions to problems with Isaac ROS.

### DNN and Triton Troubleshooting

Check [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_dnn_inference/blob/main/docs/troubleshooting.md) for solutions to problems with using DNN models and Triton.

## Updates

| Date       | Changes         |
| ---------- | --------------- |
| 2023-04-05 | Initial release |
