#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from plan_movement import Planner

NODE_NAME = 'plan_movement'



def main():
    rospy.init_node(NODE_NAME, anonymous=True)
    plan_movement = Planner(NODE_NAME)
    rospy.spin()

if __name__ == '__main__':
    main()
