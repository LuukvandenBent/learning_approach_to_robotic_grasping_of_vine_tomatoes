#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from detect_truss_obb import DetectTrussOBB

NODE_NAME = 'detect_truss_obb'



def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    detect_truss_obb = DetectTrussOBB(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()
