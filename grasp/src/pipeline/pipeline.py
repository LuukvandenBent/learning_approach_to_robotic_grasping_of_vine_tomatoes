#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import smach
import smach_ros
import os
import copy
import numpy as np

import cv2
import pyrealsense2 as rs
import ctypes
import struct

from tf.transformations import quaternion_from_euler
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
import sensor_msgs.point_cloud2
from grasp.srv import pipeline_command, detect_truss_command, set_truss_data_command, set_grasp_pose_command, create_map_command, find_grasp_candidates_command, choose_grasp_pose_from_candidates_command


NODE_NAME = 'pipeline'

class Idle(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['failure', 'move', 'detect_obb', 'detect_manual', 'map', 'grasp_candidates_manual', 'grasp_candidates_oriented_keypoint', 'grasp_pose_from_candidates'],
                             input_keys=['success', 'prev_command', 'truss_data', 'map', 'grasp_candidates', 'grasp_pose'],
                             output_keys=['command', 'prev_command', 'truss_data', 'map', 'grasp_candidates', 'grasp_pose'])

        self.node_name = NODE_NAME
        self.rqt_service = rospy.Service('rqt_service', pipeline_command, self.set_next_state)
        self.next_state = None
        self.full_pipeline = ["detect_truss_obb", "go_to_truss", "create_map", "determine_grasp_candidates_oriented_keypoint", "choose_grasp_pose_from_candidates_center", "grasp", "go_to_center", "open_gripper", "check_grasp_success", "go_to_saved_pose"]
        self.full_pipeline_crate = ["detect_truss_obb", "go_to_truss", "create_map", "determine_grasp_candidates_oriented_keypoint", "choose_grasp_pose_from_candidates_center", "grasp", "go_to_place_above", "go_to_place", "open_gripper", "check_grasp_success", "go_to_place_retreat", "go_to_saved_pose"]
        self.execute_full_pipeline = False
        self.execute_full_pipeline_from_start = False
        self.execute_full_pipeline_crate = False
        self.execute_full_pipeline_crate_from_start = False   
        self.execute_full_pipeline_repeat = False                 
    
    def set_next_state(self, command):
        print("NEXT STATE :", command)
        if command.command == "execute_full_pipeline":
            self.execute_full_pipeline = True
            self.execute_full_pipeline_from_start = True
        elif command.command == "execute_full_pipeline_crate":
            self.execute_full_pipeline_crate = True
            self.execute_full_pipeline_crate_from_start = True
        elif command.command == "toggle_execute_full_pipeline_repeat":
            self.execute_full_pipeline_repeat = not self.execute_full_pipeline_repeat
        else:
            self.next_state = command.command
        return "success"
    
    def execute(self, userdata):
        while self.next_state is None:#Wait untill a new state is set
            if self.execute_full_pipeline:#Logic for automatically doing the pipeline
                if not userdata.success:
                    self.next_state = self.full_pipeline[-1]#if something went wrong during the full pipeline, go back to saved pose
                    if self.execute_full_pipeline_repeat:
                        self.execute_full_pipeline_from_start = True
                    else:
                        self.execute_full_pipeline = False
                        self.execute_full_pipeline_from_start = False
                    break
                index = 0#Start at index 0
                if not self.execute_full_pipeline_from_start:
                    if userdata.prev_command in self.full_pipeline:#If a previous state was part of the pipeline, start there
                        index = self.full_pipeline.index(userdata.prev_command)+1
                    if index == len(self.full_pipeline):#if last action in pipeline
                        if self.execute_full_pipeline_repeat:#If repeat start again
                            index = 0
                        else:#Else stop
                            index = None
                            self.execute_full_pipeline = False
                            userdata.prev_command = None
                self.execute_full_pipeline_from_start = False
                if index is not None:
                    self.next_state = self.full_pipeline[index]
                    print("NEXT STATE :", self.next_state)
            
            elif self.execute_full_pipeline_crate:#Logic for automatically doing the pipeline
                if not userdata.success:
                    self.next_state = self.full_pipeline_crate[-1]#if something went wrong during the full pipeline, go back to saved pose
                    if self.execute_full_pipeline_repeat:
                        self.execute_full_pipeline_crate_from_start = True
                    else:
                        self.execute_full_pipeline_crate = False
                        self.execute_full_pipeline_crate_from_start = False
                    break
                index = 0#Start at index 0
                if not self.execute_full_pipeline_crate_from_start:
                    if userdata.prev_command in self.full_pipeline_crate:#If a previous state was part of the pipeline, start there
                        index = self.full_pipeline_crate.index(userdata.prev_command)+1
                    if index == len(self.full_pipeline_crate):#if last action in pipeline
                        if self.execute_full_pipeline_repeat:#If repeat start again
                            index = 0
                        else:#Else stop
                            index = None
                            self.execute_full_pipeline_crate = False
                            userdata.prev_command = None
                self.execute_full_pipeline_crate_from_start = False
                if index is not None:
                    self.next_state = self.full_pipeline_crate[index]
                    print("NEXT STATE :", self.next_state)
            rospy.sleep(0.1)#Sleep to allow time for different things
        next_state = copy.deepcopy(self.next_state)
        self.next_state = None
        userdata.command = next_state

        if (next_state == "create_crate" or next_state == "save_pose" or next_state == "save_pose_place" or next_state == "go_to_saved_pose" or next_state == "go_to_center" 
            or next_state == "go_to_place_above" or next_state == "go_to_place" or next_state == "go_to_place_retreat" or next_state == "go_to_truss"
            or next_state == "grasp" or next_state == "check_grasp_success" or next_state == "open_gripper" or next_state == "pre_grasp_gripper" or next_state == "close_gripper"):
            return 'move'
        elif next_state == "detect_truss_obb":
            return 'detect_obb'
        elif next_state == "detect_truss_manual":
            return 'detect_manual'
        elif next_state == "create_map":
            return 'map'
        elif next_state == "determine_grasp_candidates_manual":
            return 'grasp_candidates_manual'
        elif next_state == "determine_grasp_candidates_oriented_keypoint":
            return 'grasp_candidates_oriented_keypoint'
        elif (next_state == 'choose_grasp_pose_from_candidates_random' or next_state == 'choose_grasp_pose_from_candidates_center'
              or next_state == 'choose_grasp_pose_from_candidates_depth_image'):
            return 'grasp_pose_from_candidates'
        else:
            return 'failure'
class DetectObjectOBB(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command'], 
                             output_keys=['success', 'prev_command', 'truss_data'])
        try:
            rospy.wait_for_service('detect_truss_obb', timeout=30)
            self.detect_truss_obb_service = rospy.ServiceProxy('detect_truss_obb', detect_truss_command)
        except:
            print("detect_truss_obb FAILED")

    def execute(self, userdata):
        rospy.logdebug('Executing state Detect OBB')
        
        # command node
        if userdata.command == 'detect_truss_obb':
            try:
                truss_data = self.detect_truss_obb_service('detect_truss_obb').poses
            except:
                print("PIPELINE FAILED TO RETREIVE TRUSS DATA")
                userdata.success = False
                return 'failure'
            userdata.truss_data = truss_data
            userdata.prev_command = userdata.command
            userdata.success = True
            return 'success'
        userdata.success = False
        return 'failure'

class DetectObjectManual(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command'], 
                             output_keys=['success', 'prev_command', 'truss_data'])
        try:
            rospy.wait_for_service('detect_truss_manual', timeout=30)
            self.detect_truss_obb_service = rospy.ServiceProxy('detect_truss_manual', detect_truss_command)
        except:
            print("detect_truss_manual FAILED")

    def execute(self, userdata):
        rospy.logdebug('Executing state Detect Manual')
        
        # command node
        if userdata.command == 'detect_truss_manual':
            try:
                truss_data = self.detect_truss_obb_service('detect_truss_manual').poses
            except:
                print("PIPELINE FAILED TO RETREIVE TRUSS DATA")
                userdata.success = False
                return 'failure'
            userdata.truss_data = truss_data
            userdata.prev_command = userdata.command
            userdata.success = True
            return 'success'
        userdata.success = False
        return 'failure'

class GenerateMap(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command', 'truss_data'], 
                             output_keys=['success', 'prev_command', 'map'])
        try:
            rospy.wait_for_service('create_map', timeout=30)
            self.create_map_service = rospy.ServiceProxy('create_map', create_map_command)
        except:
            print("create_map_service FAILED")

    def execute(self, userdata):
        rospy.logdebug('Executing state Detect')
        
        # command node
        if userdata.command == 'create_map':
            try:
                map = self.create_map_service(userdata.truss_data)
            except:
                print("PIPELINE FAILED TO RETREIVE MAP")
                userdata.success = False
                return 'failure'
            userdata.map = map.map
            userdata.prev_command = userdata.command
            userdata.success = True
            return 'success'
        userdata.success = False
        return 'failure'

class DetermineGraspCandidatesManual(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command', 'map'], 
                             output_keys=['success', 'prev_command', 'grasp_candidates'])
        try:
            rospy.wait_for_service('find_grasp_candidates_manual', timeout=30)
            self.find_grasp_candidates_keypoints_service = rospy.ServiceProxy('find_grasp_candidates_manual', find_grasp_candidates_command)
        except:
            print("find_grasp_candidates_manual_services FAILED")

    def execute(self, userdata):
        if userdata.command == 'determine_grasp_candidates_manual':
            try:
                grasp_candidates = self.find_grasp_candidates_keypoints_service(userdata.map)
            except:
                print("PIPELINE FAILED TO FIND GRASP CANDIDATES")
                userdata.success = False
                return 'failure'
            userdata.grasp_candidates = grasp_candidates.grasp_candidates
            userdata.prev_command = userdata.command
            userdata.success = True
            return 'success'
        userdata.success = False
        return 'failure'

class DetermineGraspCandidatesOrientedKeypoint(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command', 'map'], 
                             output_keys=['success', 'prev_command', 'grasp_candidates'])
        try:
            rospy.wait_for_service('find_grasp_candidates_oriented_keypoint', timeout=30)
            self.find_grasp_candidates_oriented_keypoint_service = rospy.ServiceProxy('find_grasp_candidates_oriented_keypoint', find_grasp_candidates_command)
        except:
            print("find_grasp_candidates_oriented_keypoint_services FAILED")
    
    def execute(self, userdata):
        if userdata.command == 'determine_grasp_candidates_oriented_keypoint':
            try:
                grasp_candidates = self.find_grasp_candidates_oriented_keypoint_service(userdata.map)
            except:
                print("PIPELINE FAILED TO FIND GRASP CANDIDATES")
                userdata.success = False
                return 'failure'
            userdata.grasp_candidates = grasp_candidates.grasp_candidates
            userdata.prev_command = userdata.command
            userdata.success = True
            return 'success'
        userdata.success = False
        return 'failure'

class ChooseGraspPoseFromCandidates(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command', 'map', 'grasp_candidates', 'truss_data'], 
                             output_keys=['success', 'prev_command', 'grasp_pose'])
        try:
            rospy.wait_for_service('choose_grasp_pose_from_candidates', timeout=30)
            self.choose_grasp_pose_from_candidates_service = rospy.ServiceProxy('choose_grasp_pose_from_candidates', choose_grasp_pose_from_candidates_command)
        except:
            print("choose_grasp_pose_from_candidates_services FAILED")
    
    def execute(self, userdata):
        if userdata.command == 'choose_grasp_pose_from_candidates_random':
            command = "random"
        elif userdata.command == 'choose_grasp_pose_from_candidates_center':
            command = "center"
        elif userdata.command == 'choose_grasp_pose_from_candidates_pointcloud':
            command = "pointcloud"
        elif userdata.command == 'choose_grasp_pose_from_candidates_depth_image':
            command = "depth_image"
        try:
            truss_center = PoseStamped()
            truss_center.header = userdata.truss_data.header
            truss_center.pose = userdata.truss_data.poses[0]
            grasp_pose = self.choose_grasp_pose_from_candidates_service(command, truss_center, userdata.grasp_candidates, userdata.map)
        except:
            print("PIPELINE FAILED TO FIND GRASP CANDIDATES")
            userdata.success = False
            return 'failure'
        userdata.grasp_pose = grasp_pose.grasp_pose
        userdata.prev_command = userdata.command
        userdata.success = True
        return 'success'

class MoveRobot(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], 
                             input_keys=['command', 'truss_data', 'grasp_pose'], 
                             output_keys=['success', 'prev_command'])
        try:
            rospy.wait_for_service('move_robot', timeout=30)
            self.move_robot_service = rospy.ServiceProxy('move_robot', pipeline_command)
        except:
            print("move_robot_services FAILED")
        try:
            rospy.wait_for_service('set_truss_data', timeout=2)
            self.set_truss_data_service = rospy.ServiceProxy('set_truss_data', set_truss_data_command)
        except:
            print("set_truss_data_services FAILED")
        try:
            rospy.wait_for_service('set_grasp_pose', timeout=2)
            self.set_grasp_pose_service = rospy.ServiceProxy('set_grasp_pose', set_grasp_pose_command)
        except:
            print("set_grasp_pose_services FAILED")


    def execute(self, userdata):
        rospy.logdebug('Executing state Move Robot, command: ', userdata.command)
        possible_commands = ["create_crate", "save_pose", "save_pose_place", "go_to_saved_pose", "go_to_truss", "grasp", "go_to_center", "go_to_place_above", "go_to_place", "go_to_place_retreat", "check_grasp_success", "open_gripper", "pre_grasp_gripper","close_gripper"]
        if userdata.command in possible_commands:
            #Set relevant data
            if userdata.command == "go_to_truss":
                try:
                    self.set_truss_data_service(userdata.truss_data)
                except:
                    print("PIPELINE FAILED TO SET RELEVANT DATA")
            elif userdata.command == "grasp":
                try:
                    self.set_grasp_pose_service(userdata.grasp_pose)
                except:
                    print("PIPELINE FAILED TO SET RELEVANT DATA")
            #Move the robot
            try:
                result = self.move_robot_service(userdata.command)
                userdata.prev_command = userdata.command
                if result.success == "success":
                    userdata.success = True
                    return 'success'
                else:
                    userdata.success = False
                    return 'failure'
            except:
                print("PIPELINE FAILED TO MOVE ROBOT")
                return 'failure'
        else:
            userdata.success = False
            return 'failure'
        

        
#MAIN
def main():
    rospy.init_node('pipeline',anonymous=True)

    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['success', 'failure'])
    sis = smach_ros.IntrospectionServer('SateMachineGraphic', sm, '/SM_ROOT')#Used for smach state viewer
    sis.start()
    

    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('Idle', Idle(),
                               transitions={'detect_obb':'DetectObjectOBB',
                                            'detect_manual':'DetectObjectManual',
                                            'move': 'MoveRobot',
                                            'map':'GenerateMap',
                                            'grasp_candidates_manual':'DetermineGraspCandidatesManual',
                                            'grasp_candidates_oriented_keypoint':'DetermineGraspCandidatesOrientedKeypoint',
                                            'grasp_pose_from_candidates':'ChooseGraspPoseFromCandidates',
                                            'failure': 'Idle'})
        smach.StateMachine.add('DetectObjectOBB', DetectObjectOBB(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('DetectObjectManual', DetectObjectManual(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('GenerateMap', GenerateMap(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('DetermineGraspCandidatesManual', DetermineGraspCandidatesManual(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('DetermineGraspCandidatesOrientedKeypoint', DetermineGraspCandidatesOrientedKeypoint(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('ChooseGraspPoseFromCandidates', ChooseGraspPoseFromCandidates(),
                               transitions={'success': 'Idle',
                                            'failure': 'Idle',})
        smach.StateMachine.add('MoveRobot', MoveRobot(),
                               transitions={'success':'Idle',
                                            'failure': 'Idle'})
                                           

    # Execute SMACH plan
    sm.userdata.prev_command = None
    sm.userdata.success = True
    sm.execute()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass