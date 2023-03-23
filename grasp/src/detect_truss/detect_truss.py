#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
import tensorflow
from tensorflow import keras
from util import get_detection_model, predict_truss, create_bboxed_images, camera_info2rs_intrinsics, DepthImageFilter
from grasp.srv import detect_truss_command, detect_truss_commandResponse
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import copy

np.random.seed(16)
tensorflow.random.set_seed(16)

class DetectTruss():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.r = rospy.Rate(10)
        self.detection_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/')#todo
        self.bridge = CvBridge()
        self.image = None
        self.depth_image = None
        self.camera_info = None
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.image_callback)
        self.depth_image_sub = rospy.Subscriber("camera/depth/image_rect_raw", Image, self.depth_image_callback)
        self.detect_truss_service = rospy.Service('detect_truss', detect_truss_command, self.execute_command)
        
        self.truss_detection_pub = rospy.Publisher('truss_detection', Image, queue_size=1, latch=True)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')

        self.collect_image = False
        self.collect_depth_image = False
        self.images_saved = 0
    
    def image_callback(self, image):
        if self.collect_image:
            self.image = image
            self.collect_image = False
            self.images_saved += 1
        else:
            pass
    
    def depth_image_callback(self, depth_image):
        if self.collect_depth_image:
            self.depth_image = depth_image
            self.collect_depth_image = False
            self.images_saved += 1
        else:
            pass
    
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
    
    def execute_command(self, command):
        print("SERVICE ARIVED WITH COMMAND: ", command)
        if command.command == "detect_truss":
            self.images_saved = 0
            self.collect_image, self.collect_depth_image = True, True
            counter = 0
            while self.images_saved < 2:#Save both rgb and depth 
                rospy.sleep(0.1)
                counter += 1
                if counter > 10:
                    print("No images comming in")
                    return None
            truss_data = self.detect_truss(image=self.image, depth_image=self.depth_image)
            return truss_data
        else:
            return None

    def detect_truss(self, image=None, depth_image=None):
        rgb_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_image)
        inference_model = get_detection_model(pwd_model=self.detection_model_path)
        num_detections, bboxes_pred = predict_truss(rgb_image, inference_model)

        cropped_images, bboxes = create_bboxed_images(rgb_image, 
                                                      bboxes_pred, 
                                                      desired_size=510)
        
        
        print(f'{len(cropped_images)} DETECTIONS')

        if len(cropped_images) == 0:
            return PoseArray()

        #self.color_image_bboxed = np.array(cropped_images[0])
        xyz_mid_point, xyz_corner1, xyz_corner2 = self.generate_truss_data(image=rgb_image, depth_image=depth_image, bbox=bboxes[0])
        truss_data = PoseArray()
        truss_data.header.frame_id = image.header.frame_id
        truss_center, truss_edge1, truss_edge2 = Pose(), Pose(), Pose()
        truss_center.position.x, truss_center.position.y, truss_center.position.z = xyz_mid_point[0], xyz_mid_point[1], xyz_mid_point[2]
        
        truss_edge1.position.x, truss_edge1.position.y, truss_edge1.position.z = xyz_corner1[0], xyz_corner1[1], xyz_corner1[2]
        truss_edge2.position.x, truss_edge2.position.y, truss_edge2.position.z = xyz_corner2[0], xyz_corner2[1], xyz_corner2[2]
        truss_data.poses.extend([truss_center, truss_edge1, truss_edge2])
        truss_data = self.transform_pose_array(truss_data, self.planning_frame)

        orientation_array = quaternion_from_euler(-np.pi, np.pi/2,0)#Orientation with respect to fake_camera_bottom_screw RPY: -pi, pi/2 0
        truss_data.poses[0].orientation.x, truss_data.poses[0].orientation.y, truss_data.poses[0].orientation.z, truss_data.poses[0].orientation.w  = orientation_array[0], orientation_array[1], orientation_array[2], orientation_array[3]
        print("Truss found at :", truss_data.poses[0])
        return truss_data
    
    def generate_truss_data(self, image=None, depth_image=None, bbox=None):
        rs_intrinsics = camera_info2rs_intrinsics(self.camera_info)
        
        #Make sure bbox is not outside of image size
        bbox[0] = int(np.clip(bbox[0], 0, np.shape(image)[1]))#X
        bbox[2] = int(np.clip(bbox[2], 0, np.shape(image)[1]))#X
        bbox[1] = int(np.clip(bbox[1], 0, np.shape(image)[0]))#Y
        bbox[3] = int(np.clip(bbox[3], 0, np.shape(image)[0]))#Y

        mid_point = [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)]
        corner1 = [bbox[0], bbox[1]]
        corner2 = [bbox[2], bbox[3]]
        
        depth_image_filter = DepthImageFilter(depth_image, rs_intrinsics, patch_size=30, node_name=self.node_name)
        depth = depth_image_filter.get_depth(int(mid_point[1]), int(mid_point[0]))#Assume same depth for the corners
        depth = depth/1000#Convert from mm to m

        xyz_mid_point = depth_image_filter.deproject(int(mid_point[1]), int(mid_point[0]), depth=depth)
        xyz_corner1 = depth_image_filter.deproject(int(corner1[1]), int(corner1[0]), depth=depth)
        xyz_corner2 = depth_image_filter.deproject(int(corner2[1]), int(corner2[0]), depth=depth)

        debug_image = copy.deepcopy(image)
        debug_image = cv2.rectangle(debug_image, tuple(corner1), tuple(corner2), (255,255,255), 5)
        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        self.truss_detection_pub.publish(ros_image_msg)
        return [xyz_mid_point, xyz_corner1, xyz_corner2]
    
    def find_transform(self, source, target):
        try:
            transform = self.tfBuffer.lookup_transform(target,
                                        source,
                                       rospy.Time(0),
                                       rospy.Duration(3.0))
            return transform
        except Exception as e:
            print(e)
            return None
        
        
    def transform_pose_array(self, data, frame):
        transform = self.find_transform(data.header.frame_id, frame)
        if transform == None:
            return None
        else:
            data_copy = copy.deepcopy(data)
            for i, pose in enumerate(data_copy.poses):
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = data_copy.header.frame_id
                pose_stamped.pose = pose
                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
                data.poses[i] = transformed_pose.pose
                data.header.frame_id = frame
            return data