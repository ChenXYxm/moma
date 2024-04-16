# MoMa

This repository is created based on [the main repository](https://github.com/ethz-asl/moma) of the mobile manipulation (moma) team at ASL containing launch files, robot descriptions, utilities, controllers, and documentation for robot operation.

This repository is primarily used to apply a proposed pushing policy to a robotic arm.

## Packages

- [`moma_bringup`](moma_bringup/README.md): Launch files and configurations for interfacing with the real robots.
- [`moma_description`](moma_description/README.md): Unified Description Formats (URDFs) of our robots.
- [`moma_gazebo`](moma_gazebo/README.md): Launch files, virtual worlds and plugins for Gazebo simulation (work in progress).
- [`moma_utils`](moma_utils/README.md): Python utilities commonly used within moma projects.
- `panda_control`: Wrappers and custom controllers for controlling panda.
- `panda_moveit_config`: MoveIt configuration for our panda setup.
- [`pushing`](pushing/README.md): apply the PPO with CNN pushing method and baseline method on the panda arm.

## Quick Start

Note that all instructions in this repository are tested on Ubuntu 18.04 with ROS melodic.

First, clone this repository and its submodules into the `src` folder of a new or existing catkin workspace.

```bash
mkdir -p ~/moma_ws/src
cd ~/moma_ws/src
git clone --recurse-submodules git@github.com:ChenXYxm/moma_ws.git
```

Start up the docker
```bash
cd ~/moma_ws/src/moma/docker/
./run_docker.sh -b dev.Dockerfile
```

Set up and build the workspace

```bash
cd ~/moma_ws
catkin init
catkin config --extend /root/moma_dep_ws/devel
catkin config --cmake-args -DCMAKE_BUILD_TYPE=Release
catkin build --continue

source ~/.bashrc 
```


Install additional python libraries:
```
pip install stable-baselines3
pip install opencv-python
pip install open3d
pip install shapely
pip install scipy
```

## Download the PPO with CNN pushing model

please down load the [trained model](https://drive.google.com/drive/folders/1Cs4M6IC1g8I4HtM5DW9w-0GQS64BWv6l?usp=sharing), and save the model under the 'data' directory.

## Table calibration

Please calibrate the table following the [instructions](https://github.com/ethz-asl/moma_docs/wiki/Panda-Software-Quickstart#step-1-simulation), before applying the next step.

## apply pushing methods

please check the [`pushing`](pushing/README.md) to run the code.

