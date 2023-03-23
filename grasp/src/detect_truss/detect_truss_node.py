#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from detect_truss import DetectTruss
import os

#FORCE ON CPU, SINCE THIS METHOD IS OLD AND OTHERWISE USE GPU MEMORY
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


NODE_NAME = 'detect_truss'



def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    detect_truss = DetectTruss(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()
