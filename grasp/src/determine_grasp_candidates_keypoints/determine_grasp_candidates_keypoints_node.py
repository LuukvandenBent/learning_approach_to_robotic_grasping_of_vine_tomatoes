#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from determine_grasp_candidates_keypoints import DetermineGraspCandidatesKeypoints

NODE_NAME = 'determine_grasp_candidates_keypoints'



def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    determine_grasp_candidates_keypoints = DetermineGraspCandidatesKeypoints(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()
