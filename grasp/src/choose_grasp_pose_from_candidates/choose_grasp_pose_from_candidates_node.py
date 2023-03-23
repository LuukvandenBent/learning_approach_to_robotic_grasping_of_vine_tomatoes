#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from choose_grasp_pose_from_candidates import ChooseGraspPoseFromCandidates
import os

NODE_NAME = 'detect_truss'

def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    choose_grasp_pose_from_candidates = ChooseGraspPoseFromCandidates(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()