import rospy
import os
import numpy as np
import torch
import cv2
import copy
from cv_bridge import CvBridge
import pyrealsense2 as rs
import pcl_ros

import tf2_ros
import tf2_geometry_msgs
import tf2_sensor_msgs

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from grasp.srv import find_grasp_candidates_command
from common.transforms import find_transform, transform_pose, transform_pose_array
from common.util import camera_info2rs_intrinsics, pointcloud2numpy, pointcloud2image

class DetermineGraspCandidatesOrientedKeypoint():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.detection_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'weights/best.pt')#todo
        self.bridge = CvBridge()
        self.rs_intrinsics = None
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')
        self.camera_frame = rospy.get_param('/camera_frame')
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.grasp_candidates_raw_debug_pub = rospy.Publisher('grasp_candidates_raw_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_debug_pub = rospy.Publisher('grasp_candidates_debug', Image, queue_size=1, latch=True)
        self.grasp_candidates_decoded_debug_pub = rospy.Publisher('grasp_candidates_decoded_debug', Image, queue_size=1, latch=True)
        self.find_grasp_candidates_oriented_keypoint_services = rospy.Service('find_grasp_candidates_oriented_keypoint', find_grasp_candidates_command, self.execute_command)
        
        self.model = torch.hub.load(os.path.dirname(os.path.realpath(__file__)), 'custom', path_or_model=self.detection_model_path, source='local', force_reload=True)
        #todo set good thresholds
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
    
    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)
    
    def execute_command(self, map):
        print("CALCULATING GRASP CANDIDATES")
        try:
            grasp_candidates = self.determine_grasp_candidates_oriented_keypoint(pointcloud=map.map)
            return grasp_candidates
        except Exception as e:
            print(e)
            return None
    
    def determine_grasp_candidates_oriented_keypoint(self, pointcloud):
        transform = find_transform(pointcloud.header.frame_id, self.camera_frame, self.tfBuffer)
        pointcloud_camera_frame = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
        rgb_image, projection_history, numpy_x, numpy_y, numpy_z= pointcloud2image(pointcloud_camera_frame, self.rs_intrinsics)
        #rescale
        image_size = 640
        pad_size = int((rgb_image.shape[1]-rgb_image.shape[0])/2)#assume image is wider than it is high
        preprocessed_image = cv2.copyMakeBorder(rgb_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        image_scale = rgb_image.shape[1]/image_size
        preprocessed_image = cv2.resize(preprocessed_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
        #Run model
        results = self.model(preprocessed_image)#Model runs at resolution of 640
        bboxes = results.pred[0].cpu().numpy()
        
        self.publish_debug_image(debug_image=copy.deepcopy(preprocessed_image), bboxes=copy.deepcopy(bboxes))
        if len(bboxes) == 0:#No detections
            return None
        
        grasp_candidates = PoseArray()
        grasp_candidates.header.frame_id = self.camera_frame
        for bbox in bboxes:
        #invert the rescale for the bbox
            bbox[[0,1,2,3,6,7]] *= image_scale
            bbox[[1,3,7]] -= pad_size
            grasp_pose = self.generate_grasp_pose(projection_history=projection_history, bbox=bbox, numpy_x=numpy_x, numpy_y=numpy_y, numpy_z=numpy_z)
            grasp_candidates.poses.append(grasp_pose)
        grasp_candidates = transform_pose_array(grasp_candidates, self.planning_frame, self.tfBuffer)
        return grasp_candidates

    def generate_grasp_pose(self, projection_history, bbox, numpy_x, numpy_y, numpy_z):
        grasp_center = (int(bbox[6]), int(bbox[7]))
        grasp_orientation = np.arctan2(bbox[8], bbox[9]) + np.pi/2
        
        grasp_pose = Pose()
        chosen_point = np.array(grasp_center)#find closest match in projection_history and use the index to find the xyz value
        available_points = np.array(projection_history)
        index = np.sum((available_points-chosen_point)**2, axis=1, keepdims=True).argmin(axis=0)
        grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z = numpy_x[index], numpy_y[index], numpy_z[index]
        grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w = quaternion_from_euler(0, 0, grasp_orientation)
        return grasp_pose

    def publish_debug_image(self, debug_image, bboxes):
        self.grasp_candidates_raw_debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8"))
        decoded_debug_image = copy.deepcopy(debug_image)
        color_blue = (0,0,255)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1,y1,x2,y2 = bbox[:4]#Bbox
            debug_image = cv2.rectangle(debug_image, (int(x1),int(y1)), (int(x2),int(y2)), (255,255,0), 2)
            x_c,y_c = int(bbox[6]), int(bbox[7])
            angle = np.arctan2(bbox[8], bbox[9])
            p1, p2 = (int(x_c + 50*np.cos(angle)), int(y_c + 50*np.sin(angle))), (int(x_c - 50*np.cos(angle)), int(y_c - 50*np.sin(angle)))
            debug_image = cv2.circle(debug_image, (x_c,y_c), 5, color_blue, -1)
            debug_image = cv2.line(debug_image, p1, p2, color_blue, 2)
            #Draw center and orientation for decoded image

            decoded_debug_image = cv2.line(decoded_debug_image, p1, p2, color_blue, 2)
            decoded_debug_image = cv2.circle(decoded_debug_image, (x_c,y_c), 5, color_blue, -1)

        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        ros_decoded_image_msg = self.bridge.cv2_to_imgmsg(decoded_debug_image, encoding="rgb8")
        self.grasp_candidates_debug_pub.publish(ros_image_msg)
        self.grasp_candidates_decoded_debug_pub.publish(ros_decoded_image_msg)