#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from detect_truss_manual import DetectTrussManual

NODE_NAME = 'detect_truss_manual'

from threading import Lock

def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    detect_truss_manual = DetectTrussManual(NODE_NAME)
    lock = Lock()
    while True:
        rospy.sleep(1)
        with lock:
            if detect_truss_manual.draw:
                detect_truss_manual.draw_bboxes(detect_truss_manual.preprocessed_image)
                detect_truss_manual.draw = False
    #rospy.spin()

if __name__ == '__main__':
    main()
