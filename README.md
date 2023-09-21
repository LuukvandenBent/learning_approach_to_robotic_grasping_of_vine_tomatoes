# learning_approach_to_robotic_grasping_of_vine_tomatoes
This repository will contain all the code which is used in my Msc. thesis: Learning Approach To Robotic Grasping Of Vine Tomatoes

## 1. Install
> :warning: This project is built on Ubuntu 20.04 and ROS Noetic!

Install ROS [noetic](http://wiki.ros.org/noetic/Installation) (Ubuntu 20.04). Make sure that you have your environment properly setup, and that you have the most up-to-date packages:
```
rosdep update  # No sudo
sudo apt-get update
sudo apt-get dist-upgrade
```

This project is set-up in the following way: The computer in the lab has a real-time kernel installed and runs the controller for the Franka Emika Panda and will act as the ROS MASTER. Your own computer will connect to this PC and command actions through ROS.

### 1.1 LAB PC
#### 1.1.1 Create A Workspace
You will need to have a ROS workspace setup:
```
mkdir -p ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/
```
#### 1.1.2 Download the source code
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
git clone https://github.com/LuukvandenBent/franka_ros_vine_tomato.git
cd franka_ros_vine_tomato
git clone https://github.com/franzesegiovanni/franka_human_friendly_controllers.git
cd ..
git clone https://github.com/LuukvandenBent/panda_moveit_config_vine_tomato.git
git clone https://github.com/LuukvandenBent/moveit_calibration_vine_tomato.git
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
cd ../..
```
### 1.1.3 Setup Libfranka
Follow the guide for building libfranka from source for either the panda or fr3:
[Libfranka](https://frankaemika.github.io/docs/installation_linux.html)

#### 1.1.3 Build and launch
```
catkin_make -DBUILD_TESTS=OFF -DFranka_DIR:PATH=<path/to/libfranka/build>
catkin_make -DBUILD_TESTS=OFF -DFranka_DIR:PATH=~/libfranka/build 
source devel/setup.bash
roslaunch panda_moveit_config_vine_tomato launch_moveit.launch robot_ip:=<robot_ip> robot_name:=<panda or fr3>
```

### 1.2 Personal PC
Next we will set-up your own pc:
#### 1.2.1 Create A Workspace
You will need to have a ROS workspace setup:
```
mkdir -p ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/
catkin build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
```

#### 1.2.2 Download the source code

clone this repository
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
git clone https://github.com/LuukvandenBent/learning_approach_to_robotic_grasping_of_vine_tomatoes.git
```

clone some other (forked and slightly changed dependencies)
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
git clone https://github.com/LuukvandenBent/realsense-ros_d405_ros1.git
```

install some dependecies
```
sudo apt-get install ros-$ROS_DISTRO-realsense2-camera
```

#### 1.2.3 Setup ROS MASTER
To connect with the computer in the lab:
```
export ROS_MASTER_URI=http://<computer_ip>:11311 
export ROS_IP=<pc_ip> 
export ROS_HOSTNAME=<pc_ip>
```
Example:
```
export ROS_MASTER_URI=http://172.16.0.1:11311 
export ROS_IP=172.16.0.30 
export ROS_HOSTNAME=172.16.0.30
```
#### 1.2.4 Build and launch
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws
catkin build -DPCL_DIR:PATH=<path/to/pcl>
catkin build -DPCL_DIR:PATH=/usr/lib/x86_64-linux-gnu/cmake/pcl
source devel/setup.bash
roslaunch grasp vine_grasping.launch robot_name:=<panda or fr3>
```

#### 1.2.5 Installing packages
The Object and Pose Detection networks run on GPU. For this to work we need Pytorch with the correct CUDA version. This is tested with CUDA 11.3 but other versions should work.
Ensure that the Nvidia Compiler and driver are (almost) the same version. nvidia-smi (driver) can have a slightly higher version than NVCC -V (compiler), i.e. 11.4 and 11.3. 
```
nvidia-smi
NVCC -V
```
Install PyTorch with the corresponding CUDA version, for example: 
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then run:
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws
cd src/learning_approach_to_robotic_grasping_of_vine_tomatoes/grasp/src/detect_truss_obb/utils/nms_rotated
pip install -v -e .
```
If a warning appears about THC.h or ATen.h, please see this [commit](https://github.com/hukaixuan19970627/yolov5_obb/commit/622026fd72ccad330fa1c4cc98774d49a0fd8401), change the file 
utils/nms_rotated/src/poly_nms_cuda.cu depending on the error. This is caused by changing libraries in newer CUDA versions.  


> :warning: When running for the first time, the camera pose relative to the arm has to be calibrated, this should be done using the [Moveit Hand-eye Calibration](https://ros-planning.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html). Run this from the Lab-PC and save in: "moveit_calibration_vine_tomato/moveit_calibration_gui/camera_calibration.launch"

