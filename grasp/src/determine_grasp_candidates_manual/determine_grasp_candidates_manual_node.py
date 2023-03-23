#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from determine_grasp_candidates_manual import DetermineGraspCandidatesManual
from threading import Lock

NODE_NAME = 'determine_grasp_candidates_manual'

def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    determine_grasp_candidates_manual = DetermineGraspCandidatesManual(NODE_NAME)
    lock = Lock()
    while True:
        rospy.sleep(1)
        with lock:
            if determine_grasp_candidates_manual.draw:
                determine_grasp_candidates_manual.draw_poses(determine_grasp_candidates_manual.preprocessed_image)
                determine_grasp_candidates_manual.draw = False

if __name__ == '__main__':
    main()
