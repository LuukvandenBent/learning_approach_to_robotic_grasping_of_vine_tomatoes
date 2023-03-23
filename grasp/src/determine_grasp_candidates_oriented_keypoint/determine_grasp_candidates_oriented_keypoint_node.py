#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from determine_grasp_candidates_oriented_keypoint import DetermineGraspCandidatesOrientedKeypoint

NODE_NAME = 'determine_grasp_candidates_oriented_keypoint'



def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    determine_grasp_candidates_oriented_keypoint = DetermineGraspCandidatesOrientedKeypoint(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()
