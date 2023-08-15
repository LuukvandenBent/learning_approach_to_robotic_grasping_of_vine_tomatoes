# learning_approach_to_robotic_grasping_of_vine_tomatoes
This repository will contain all the code which is used in my Msc. thesis: Learning Approach To Robotic Grasping Of Vine Tomatoes

## 1. Install
> :warning: This project is built on Ubuntu 20.04 and ROS Noetic!

Install ROS [noetic](http://wiki.ros.org/noetic/Installation) (Ubuntu 20.04). Make sure sure that you have your environment properly setup, and that you have the most up to date packages:
```
rosdep update  # No sudo
sudo apt-get update
sudo apt-get dist-upgrade
```

This project is set-up in the following way: The computer in the lab has a real-time kernel installed and runs the controller for the Franka Emika Panda and will act as the ROS MASTER. Your own computer will connect to this pc and command actions trough ROS.

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
cd src/franka_ros_vine_tomato
git clone https://github.com/franzesegiovanni/franka_human_friendly_controllers.git
cd ..
git clone https://github.com/LuukvandenBent/panda_moveit_config_vine_tomato.git
git clone https://github.com/LuukvandenBent/moveit_calibration_vine_tomato.git
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1`
cd ../..
```
#### 1.1.3 Build and launch
```
catkin_make -DBUILD_TESTS=OFF -DFranka_DIR:PATH=<path/to/libfranka/build>
catkin_make -DBUILD_TESTS=OFF -DFranka_DIR:PATH=~/libfranka/build 
source/devel/setup.bash
roslaunch panda_moveit_config_vine_tomato launch_moveit.launch robot_ip:=<robot_ip>
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

clone some other (forked and slightly changed dependecies)
```
cd ~/learning_approach_to_robotic_grasping_of_vine_tomatoes_ws/src
git clone https://github.com/LuukvandenBent/realsense-ros_d405_ros1.git
git clone https://github.com/pal-robotics/aruco_ros.git
git clone https://github.com/IFL-CAMP/easy_handeye.git
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
catkin build -DPCL_DIR:PATH=<path/to/pcl>
catkin build -DPCL_DIR:PATH=/usr/lib/x86_64-linux-gnu/cmake/pcl
source devel/setup.bash
roslaunch grasp vine_grasping.launch
```

> :warning: When running for the first time, the camera pose relative to the arm has to be calibrated, see package 'Cailbration'

