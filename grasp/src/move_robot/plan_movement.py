#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import rospkg
import sys
import shutil
import numpy as np
import quaternion
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, WrenchStamped
from std_msgs.msg import Float32MultiArray, Float32, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import moveit_commander
import moveit_msgs.msg
import dynamic_reconfigure.client

from grasp.srv import pipeline_command, pipeline_commandResponse, set_truss_data_command, set_truss_data_commandResponse, set_grasp_pose_command, set_grasp_pose_commandResponse
from controller_manager_msgs.srv import SwitchController
from common.transforms import transform_pose
from common.moveit_util import all_close, create_collision_object
from common.util import flip_z_rotation_stamped_pose

import tf2_ros
import tf2_geometry_msgs

import actionlib
import franka_gripper.msg
from pilz_robot_programming import *

import copy
import os
class Planner(object):

    def __init__(self, NODE_NAME):
        super(Planner, self).__init__()
        self.node_name = NODE_NAME

        self.vine_radius = 0.0025
        self.succesfull_grasp_force_difference = 0.3#0.15 for fake
        self.force_limit = 6#Newton
        
        self.crate_width, self.crate_length, self.crate_height = 0.36, 0.56, 0.23
        
        self.grasp_pose = None
        self.truss_pose = None
        self.aruco1_pose = None
        self.aruco2_pose = None
        self.controller_running = None
        self.force_z = None

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)#maybe unused
        self.planning_frame = rospy.get_param('/planning_frame')
        self.camera_frame = rospy.get_param('/camera_frame')
        self.robot_name = rospy.get_param('/robot_name')

        self.aruco1_pose_sub = rospy.Subscriber("aruco_tracker/pose", Pose, self.aruco1_pose_callback)
        self.aruco2_pose_sub = rospy.Subscriber("aruco_tracker/pose2", Pose, self.aruco2_pose_callback)
        
        self.cartesian_pose_pub = rospy.Publisher("/equilibrium_pose", PoseStamped, queue_size=0)
        self.force_ext_sub = rospy.Subscriber("franka_state_controller/F_ext", WrenchStamped, self.force_ext_callback, queue_size=1)
        
        self.move_robot_service = rospy.Service('move_robot', pipeline_command, self.plan_movement)
        self.set_truss_data_service = rospy.Service('set_truss_data', set_truss_data_command, self.set_truss_pose)
        self.set_grasp_pose_service = rospy.Service('set_grasp_pose', set_grasp_pose_command, self.set_grasp_pose)
        self.switch_controller_service = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
        
        #MOVEIT SETUP
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher("/move_group/display_planned_path", moveit_msgs.msg.DisplayTrajectory, queue_size=20)

        self.move_group = moveit_commander.MoveGroupCommander(self.robot_name+"_arm")
        self.move_group_ee = moveit_commander.MoveGroupCommander(self.robot_name+"_manipulator")
        self.move_group_ee.set_end_effector_link(self.robot_name+'_hand_tcp')
        self.move_group_camera = moveit_commander.MoveGroupCommander(self.robot_name+"_camera")

        self.gripper_grasp_action = actionlib.SimpleActionClient('franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.gripper_move_action = actionlib.SimpleActionClient('franka_gripper/move', franka_gripper.msg.MoveAction)
    
        rospy.sleep(2)#Needed
        ceiling = create_collision_object(robot=self.robot, id='ceiling', dimensions=[2, 2, 0.02], pose=[0, 0, 1])#Add limits for table and wall
        wall1 = create_collision_object(robot=self.robot, id='wall1', dimensions=[0.02, 1, 1], pose=[-0.5, 0, 0.5], orientation=[0,0,0])#Add limits for table and wall
        wall2 = create_collision_object(robot=self.robot, id='wall2', dimensions=[0.02, 1, 1], pose=[0, -0.2, 0.5], orientation=[0,0,np.pi/2])#Add limits for table and wall

        self.scene.add_object(ceiling)
        self.scene.add_object(wall1)
        self.scene.add_object(wall2)
        
        for move_group in [self.move_group, self.move_group_ee, self.move_group_camera]:
            move_group.get_current_pose()#Call get_current_pose once to fix a bug: https://github.com/ros-planning/moveit/issues/2715
        
        command = pipeline_command()
        command.command = "save_pose"
        self.plan_movement(command)#Save pose at start

    def aruco1_pose_callback(self, data):
        if self.aruco1_pose is None:
            data_stamped = PoseStamped()
            data_stamped.header.frame_id = self.camera_frame 
            data_stamped.pose = data
            self.aruco1_pose = transform_pose(data_stamped, self.planning_frame, self.tfBuffer)

    def aruco2_pose_callback(self, data):
        if self.aruco2_pose is None:
            data_stamped = PoseStamped()
            data_stamped.header.frame_id = self.camera_frame
            data_stamped.pose = data
            self.aruco2_pose = transform_pose(data_stamped, self.planning_frame, self.tfBuffer)
    
    def force_ext_callback(self, data):
        self.force_z = data.wrench.force.z
    
    def set_truss_pose(self, data):
        data_stamped = PoseStamped()
        data_stamped.header.frame_id = data.poses.header.frame_id 
        data_stamped.pose = data.poses.poses[0]
        self.truss_pose = transform_pose(data_stamped, self.planning_frame, self.tfBuffer)
        return 'success'
    
    def set_grasp_pose(self, data):
        self.grasp_pose = transform_pose(data.grasp_pose, self.planning_frame, self.tfBuffer)
        return 'success'

    def plan_movement(self, data):
        movement = data.command
        print("Planning movement command: ", movement)
        
        if movement == 'save_pose':
            self.saved_pose_ee = self.move_group_ee.get_current_pose()
            self.saved_joints_ee = self.move_group_ee.get_current_joint_values()
            print('Current pose is saved')
            return "success"
        
        elif movement == 'save_pose_place':
            self.saved_pose_ee_place = self.move_group_ee.get_current_pose()
            print('Current place pose is saved')
            return "success"
        
        elif movement == 'create_crate':
            return self.create_crate_collision_object()
        
        elif movement == 'open_gripper':
            return self.open_gripper()
        
        elif movement == 'pre_grasp_gripper':
            return self.pre_grasp_gripper()
        
        elif movement == 'close_gripper':
            return self.close_gripper()
        
        elif movement == 'grasp':
            return self.grasp()
        
        elif movement == 'check_grasp_success':
            return self.check_grasp_success()

        goal, move_group, allow_flip, retry_with_impedance = self.find_goal_pose(movement=movement)

        return self.go_to_pose(goal, move_group, allow_flip=allow_flip, retry_with_impedance=retry_with_impedance)

    def find_goal_pose(self, movement=None):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = self.planning_frame
        move_group = self.move_group_ee
        allow_flip = False
        retry_with_impedance = False

        if movement == 'go_to_saved_pose':
            self.open_gripper()
            goal_pose = self.saved_pose_ee
        
        elif movement == 'go_to_center':
            goal_pose = copy.deepcopy(self.saved_pose_ee)
            goal_pose.pose.position.z = 0.1
        
        elif movement == 'go_to_place_above':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            goal_pose.pose.position.z = self.move_group_ee.get_current_pose().pose.position.z
            allow_flip = True
            retry_with_impedance = True
            
        elif movement == 'go_to_place':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            allow_flip = True
            retry_with_impedance
        
        elif movement == 'go_to_place_retreat':
            goal_pose = copy.deepcopy(self.saved_pose_ee_place)
            goal_pose.pose.position.z = goal_pose.pose.position.z + 0.25
            allow_flip = True
            retry_with_impedance
        
        elif movement == 'go_to_truss':
            move_group = self.move_group_camera
            goal_pose = self.truss_pose
            goal_pose.pose.position.z = goal_pose.pose.position.z + 0.10#Camera ~10 cm above
            allow_flip = True

        else:
            goal_pose = move_group.get_current_pose()

        return goal_pose, move_group, allow_flip, retry_with_impedance


    # control robot to desired goal position
    def go_to_pose(self, goal_pose, move_group, controller="position", allow_flip=False, retry_with_impedance=False, speed=1):#speed for impedance
        if goal_pose == None:
            return 'failure'
        
        if controller == "position":
            if self.controller_running != controller:
                switched = self.switch_controller(controller)
                if not switched:
                    return 'failure'
            move_group.set_planning_pipeline_id("pilz_industrial_motion_planner")
            move_group.set_planner_id("LIN")
            move_group.set_max_acceleration_scaling_factor(0.3)
            move_group.set_max_velocity_scaling_factor(0.7)
            move_group.set_pose_target(goal_pose)
            success, plan, _, error = move_group.plan()
            if retry_with_impedance and not success:#Try again with impedance
                print("Could not plan LIN due to: ", error, "retrying with impedance")
                return self.go_to_pose(goal_pose=goal_pose, move_group=move_group, controller="impedance", speed=10)
                    

            print("PLANNING WAS : ", success)
            if not success and goal_pose != self.saved_pose_ee:
                if allow_flip:
                    flipped_goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                    return self.go_to_pose(goal_pose=flipped_goal_pose, move_group=move_group)#retry flipped
                else:
                    return 'failure'
            elif success:
                move_group.execute(plan, wait=True)
                move_group.stop()
                move_group.clear_pose_targets()
        
        elif controller == "impedance":
            if move_group != self.move_group_ee:
                print("CARTESIAN IMPEDANCE CONTROLLER WORKS WITH RESPECT TO THE END-EFFECTOR, WRONG MOVE GROUP: ", move_group)
                return 'failure'
            if self.controller_running != controller:
                switched = self.switch_controller(controller)
                if not switched:
                    return 'failure'
                start_pose = move_group.get_current_pose()
                if not all_close(start_pose, goal_pose, 0.1, only_check_angle=True):
                    goal_pose = flip_z_rotation_stamped_pose(goal_pose)
                self.cartesian_go_to_pose(start_pose=start_pose, goal_pose=goal_pose, speed=speed)
                return 'success'
    
        else:
            print("UNKOWN CONTROLLER: ", controller)
            return 'failure'  
        
        if goal_pose == self.saved_pose_ee:#Also reset joints when going to saved pose
            move_group.set_planner_id("PTP")
            move_group.go(self.saved_joints_ee, wait=True)
            move_group.stop()
            move_group.clear_pose_targets()
        
        #Test if plan succeeded
        current_pose = move_group.get_current_pose().pose
        close = all_close(goal_pose.pose, current_pose, 0.05)
        print("MOVEMENT WAS CLOSE?: ", close)
        if close:
            return 'success'
        else:
            return 'failure'

    def display_trajectory(self, plan):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory) 

    def grasp(self):
        supplied_grasp_pose = self.grasp_pose
        if self.pre_grasp_gripper() == "success":#Open the gripper
            pre_grasp_pose = copy.deepcopy(supplied_grasp_pose)
            pre_grasp_pose.pose.position.z = pre_grasp_pose.pose.position.z + 0.02
            grasp_pose = copy.deepcopy(supplied_grasp_pose)
            grasp_pose.pose.position.z = grasp_pose.pose.position.z# - 4*self.vine_radius#Account for width of vine + little extra
            if self.go_to_pose(goal_pose=pre_grasp_pose, move_group=self.move_group_ee, allow_flip=True) == "success":#Go to pre grasp pose
                if self.go_to_pose(goal_pose=grasp_pose, move_group=self.move_group_ee, controller = "impedance") == "success":#Go to grasp pose
                    if self.close_gripper() == "success":#Grasp
                        post_grasp_pose = copy.deepcopy(supplied_grasp_pose)
                        post_grasp_pose.pose.position.z = post_grasp_pose.pose.position.z + 0.2
                        return self.go_to_pose(goal_pose=post_grasp_pose, move_group=self.move_group_ee, allow_flip=True)#retreat
        return "failure"
    
    def open_gripper(self):
        rospy.sleep(0.5)#Delay to allow force to settle
        force_data = rospy.wait_for_message("franka_state_controller/F_ext", WrenchStamped, timeout=5)
        if force_data is None:
            print("NO FORCE DATA INCOMING")
        else:
            self.ext_force_no_grasp = force_data.wrench.force.z#Set force data to compare after dropping
        self.gripper_move_action.wait_for_server()
        gripper = franka_gripper.msg.MoveGoal()
        gripper.width = 0.1
        gripper.speed = 0.05
        self.gripper_move_action.send_goal(gripper)
        self.gripper_move_action.wait_for_result()
        result =  self.gripper_move_action.get_result()
        return "success" if result else "failure"
    
    def pre_grasp_gripper(self):
        self.gripper_move_action.wait_for_server()
        gripper = franka_gripper.msg.MoveGoal()
        gripper.width = self.vine_radius*6
        gripper.speed = 0.1
        self.gripper_move_action.send_goal(gripper)
        self.gripper_move_action.wait_for_result()
        result =  self.gripper_move_action.get_result()
        return "success" if result else "failure"
    
    def close_gripper(self):
        self.gripper_grasp_action.wait_for_server()
        gripper = franka_gripper.msg.GraspGoal()
        gripper.width = 0.001
        gripper.epsilon.inner = 0.001
        gripper.epsilon.outer = 0.04
        gripper.speed = 0.005
        gripper.force = 40

        self.gripper_grasp_action.send_goal(gripper)
        self.gripper_grasp_action.wait_for_result()
        result =  self.gripper_grasp_action.get_result()
        return "success" if result else "failure"
    
    def check_grasp_success(self, save=True):
        force_data = rospy.wait_for_message("franka_state_controller/F_ext", WrenchStamped, timeout=5)
        if force_data is None:
            print("No force data coming in")
            return "failure"
        force = force_data.wrench.force.z
        result = abs(force - self.ext_force_no_grasp) > self.succesfull_grasp_force_difference
        print("FORCE BEFORE GRASP: ",self.ext_force_no_grasp, ".. FORCE AFTER GRASP: ", force, "...GRASP WAS :", result)
        #Save the pointcloud as success or failure:
        if save:
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            pointcloud_dir = os.path.join(catkin_ws_dir, "experiments/pointcloud")
            pointcloud_success_dir = os.path.join(pointcloud_dir, "success")
            pointcloud_failure_dir = os.path.join(pointcloud_dir, "failure")
            if not os.path.exists(pointcloud_success_dir):
                os.makedirs(pointcloud_success_dir)
            if not os.path.exists(pointcloud_failure_dir):
                os.makedirs(pointcloud_failure_dir)
            file_list = list()
            for file in os.listdir(pointcloud_dir):
                if file.endswith(".txt"):
                    file_list.append(file)
            if len(file_list) > 1:
                print("MORE THAN ONE UNLABELED POINTCLOUD, CHECK WHAT IS GOING ON!!!")
                return "failure"
            file_path = os.path.join(pointcloud_dir, file_list[0])
            if result:
                dest_dir = pointcloud_success_dir
            else:
                dest_dir = pointcloud_failure_dir 
            shutil.move(file_path, os.path.join(dest_dir, file_list[0]))     
        return "success"# if result else "failure" #Just return success

    def create_crate_collision_object(self):
        self.aruco1_pose, self.aruco2_pose = None, None
        counter = 0
        while self.aruco1_pose is None or self.aruco2_pose is None:
            rospy.sleep(0.2) 
            counter += 1
            if counter > 10:
                print("COULD NOT GET ARUCO POSES IN TIME")
                return "failure"
        z_min = (self.aruco1_pose.pose.position.z + self.aruco2_pose.pose.position.z)/2
        center = [(self.aruco1_pose.pose.position.x+self.aruco2_pose.pose.position.x)/2, (self.aruco1_pose.pose.position.y+self.aruco2_pose.pose.position.y)/2]
        poses = [[center[0]+self.crate_width/2, center[1], z_min+self.crate_height/2], [center[0]-self.crate_width/2, center[1], z_min+self.crate_height/2], [center[0], center[1]+self.crate_length/2, z_min+self.crate_height/2], [center[0], center[1]-self.crate_length/2, z_min+self.crate_height/2]]
        crate_1 = create_collision_object(robot=self.robot, id='crate_1', dimensions=[0.01, self.crate_length, self.crate_height], pose=poses[0])
        crate_2 = create_collision_object(robot=self.robot, id='crate_2', dimensions=[0.01, self.crate_length, self.crate_height], pose=poses[1])
        crate_3 = create_collision_object(robot=self.robot, id='crate_3', dimensions=[self.crate_width,0.01, self.crate_height], pose=poses[2])
        crate_4 = create_collision_object(robot=self.robot, id='crate_4', dimensions=[self.crate_width, 0.01, self.crate_height], pose=poses[3])

        self.scene.add_object(crate_1)
        self.scene.add_object(crate_2)
        self.scene.add_object(crate_3)
        self.scene.add_object(crate_4)
        return "success"
    
    def switch_controller(self, new_controller):
        #For the cartesian impedance controller, first set the current atractor at the current position so no sudden movements happen
        self.cartesian_pose_pub.publish(self.move_group_ee.get_current_pose())
        possible_controllers = {"position":"position_joint_trajectory_controller", "impedance":"cartesian_variable_impedance_controller"}
        start_controllers = [possible_controllers[new_controller]]
        possible_controllers.pop(new_controller)#Remove from dict to access the other controller
        stop_controllers = [next(iter(possible_controllers.values()))]
        resp = self.switch_controller_service(start_controllers=start_controllers, stop_controllers=stop_controllers, strictness=1, start_asap=False, timeout=0)
        if resp.ok:
            self.controller_running = new_controller
        return resp.ok
    
    def cartesian_go_to_pose(self, start_pose, goal_pose, speed=1):#from https://github.com/franzesegiovanni/franka_human_friendly_controllers/blob/master/python/LfD/Learning_from_demonstration.py
        assert speed < 11, "Attempt at too fast impedance control"
        start = np.array([start_pose.pose.position.x, start_pose.pose.position.y, start_pose.pose.position.z]) 
        start_ori = np.array([start_pose.pose.orientation.w, start_pose.pose.orientation.x, start_pose.pose.orientation.y, start_pose.pose.orientation.z]) 
        goal_=np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
        squared_dist = np.sum(np.subtract(start, goal_)**2, axis=0)
        dist = np.sqrt(squared_dist)
        interp_dist = 0.001*speed  # [m]
        step_num_lin = int(np.floor(dist / interp_dist))
        q_start=np.quaternion(start_ori[0], start_ori[1], start_ori[2], start_ori[3])
        q_goal=np.quaternion(goal_pose.pose.orientation.w, goal_pose.pose.orientation.x, goal_pose.pose.orientation.y, goal_pose.pose.orientation.z)
        inner_prod=q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w
        if inner_prod < 0:
            q_start.x=-q_start.x
            q_start.y=-q_start.y
            q_start.z=-q_start.z
            q_start.w=-q_start.w
        inner_prod=np.clip(q_start.x*q_goal.x+q_start.y*q_goal.y+q_start.z*q_goal.z+q_start.w*q_goal.w, -1, 1)
        theta= np.arccos(np.abs(inner_prod))
        interp_dist_polar = 0.001*speed
        step_num_polar = int(np.floor(theta / interp_dist_polar))
        step_num=np.max([step_num_polar,step_num_lin])
        
        x = np.linspace(start[0], goal_pose.pose.position.x, step_num)
        y = np.linspace(start[1], goal_pose.pose.position.y, step_num)
        z = np.linspace(start[2], goal_pose.pose.position.z, step_num)
        
        goal = PoseStamped()
        
        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]
        
        
        quat=np.slerp_vectorized(q_start, q_goal, 0.0)
        goal.pose.orientation.x = quat.x
        goal.pose.orientation.y = quat.y
        goal.pose.orientation.z = quat.z
        goal.pose.orientation.w = quat.w

        self.cartesian_pose_pub.publish(goal)
        
        if speed > 1:#normal moving
            self.set_stiffness(500, 500, 500, 10, 10, 10, 5)#XYZRPY
        else:
            self.set_stiffness(3000, 3000, 1000, 30, 30, 5, 5)#XYZRPY

        rate = rospy.Rate(20)
        goal = PoseStamped()
        contact = False
        for i in range(step_num):
            if self.force_z > self.force_limit and not contact:
                contact = True
                print("CONTACT DETECTED, NOT CHANGING THE EQUILIBRIUM POSE")
            if not contact:    
                goal.header.seq = 1
                goal.header.stamp = rospy.Time.now()
                goal.header.frame_id = self.planning_frame

                goal.pose.position.x = x[i]
                goal.pose.position.y = y[i]
                goal.pose.position.z = z[i]
                quat=np.slerp_vectorized(q_start, q_goal, i/step_num)
                goal.pose.orientation.x = quat.x
                goal.pose.orientation.y = quat.y
                goal.pose.orientation.z = quat.z
                goal.pose.orientation.w = quat.w
                self.cartesian_pose_pub.publish(goal)
            rate.sleep()
    
    def set_stiffness(self, k_t1, k_t2, k_t3,k_r1,k_r2,k_r3, k_ns):
        set_K = dynamic_reconfigure.client.Client('dynamic_reconfigure_compliance_param_node', config_callback=None)
        set_K.update_configuration({"translational_stiffness_X": k_t1})
        set_K.update_configuration({"translational_stiffness_Y": k_t2})
        set_K.update_configuration({"translational_stiffness_Z": k_t3})        
        set_K.update_configuration({"rotational_stiffness_X": k_r1}) 
        set_K.update_configuration({"rotational_stiffness_Y": k_r2}) 
        set_K.update_configuration({"rotational_stiffness_Z": k_r3})
        set_K.update_configuration({"nullspace_stiffness": k_ns})  