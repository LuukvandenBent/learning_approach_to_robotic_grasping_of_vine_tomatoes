import rospy
import os
import numpy as np
import torch
import cv2
import copy
from cv_bridge import CvBridge
import pyrealsense2 as rs
import pcl_ros
import rospkg

import tf2_ros
import tf2_geometry_msgs
import tf2_sensor_msgs

import sensor_msgs.point_cloud2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from threading import Lock
import datetime

from grasp.srv import find_grasp_candidates_command
from common.transforms import find_transform, transform_pose, transform_pose_array
from common.util import camera_info2rs_intrinsics, pointcloud2numpy

class DetermineGraspCandidatesManual():
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
        self.find_grasp_candidates_manual_services = rospy.Service('find_grasp_candidates_manual', find_grasp_candidates_command, self.execute_command)
        self.draw = False
        self.bboxes = None
        self.lock = Lock()
    
    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)
    
    def execute_command(self, map):
        print("ANNOTATE GRASP POSE(S)")
        try:
            grasp_candidates = self.determine_grasp_candidates_manual(pointcloud=map.map)
            return grasp_candidates
        except Exception as e:
            print(e)
            return None
    
    def determine_grasp_candidates_manual(self, pointcloud):
        rgb_image, projection_history, numpy_x, numpy_y, numpy_z= self.pointcloud_to_image(pointcloud)
        #rescale
        image_size = 640
        pad_size = int((rgb_image.shape[1]-rgb_image.shape[0])/2)
        preprocessed_image = cv2.copyMakeBorder(rgb_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        image_scale = rgb_image.shape[1]/image_size
        preprocessed_image = cv2.resize(preprocessed_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
        
        #Draw annotation from main thread, since opencv cant handle it otherwise
        self.preprocessed_image = preprocessed_image#Used for running in main thread
        self.draw = True
        with self.lock:#Do the drawing in the main thread
            while self.bboxes is None:
                rospy.sleep(1)
        
        bboxes = copy.deepcopy(self.bboxes)
        save_annotation = copy.deepcopy(self.save_annotation)
        self.bboxes, self.save_annotation = None, None
        self.publish_debug_image(debug_image=copy.deepcopy(preprocessed_image), bboxes=copy.deepcopy(bboxes), save=save_annotation)
        if len(bboxes) == 0:#No detections
            return None
        
        grasp_candidates = PoseArray()
        grasp_candidates.header.frame_id = self.camera_frame
        for bbox in bboxes:
            bbox[[1,2,3,4,5,6,8,9,11,12]] *= image_scale*image_size#Invert the rescale, DIFFERENT FROM determine_grasp_candidates_keypoints, since the class is first here
            bbox[[2,4,6,9,12]] -= pad_size
            grasp_pose = self.generate_grasp_pose(projection_history=projection_history, bbox=bbox, numpy_x=numpy_x, numpy_y=numpy_y, numpy_z=numpy_z)
            grasp_candidates.poses.append(grasp_pose)
        grasp_candidates = transform_pose_array(grasp_candidates, self.planning_frame, self.tfBuffer)
        return grasp_candidates

    def generate_grasp_pose(self, projection_history, bbox, numpy_x, numpy_y, numpy_z):
        grasp_center = (int(bbox[5]), int(bbox[6]))
        keypoint_left = (int(bbox[8]), int(bbox[9]))
        keypoint_right = (int(bbox[11]), int(bbox[12]))
        grasp_orientation = np.arctan2(keypoint_right[1] - keypoint_left[1], keypoint_right[0] - keypoint_left[0])
        
        grasp_pose = Pose()
        chosen_point = np.array(grasp_center)#find closest match in projection_history and use the index to find the xyz value
        available_points = np.array(projection_history)
        index = np.sum((available_points-chosen_point)**2, axis=1, keepdims=True).argmin(axis=0)
        grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z = numpy_x[index], numpy_y[index], numpy_z[index]
        grasp_pose.orientation.x, grasp_pose.orientation.y, grasp_pose.orientation.z, grasp_pose.orientation.w = quaternion_from_euler(0, 0, grasp_orientation)
        return grasp_pose
    
    def pointcloud_to_image(self, pointcloud):
        transform = find_transform(pointcloud.header.frame_id, self.camera_frame, self.tfBuffer)
        pointcloud_camera_frame = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
        reconstructed_image = np.zeros((self.rs_intrinsics.height, self.rs_intrinsics.width, 3))
        numpy_x, numpy_y, numpy_z, numpy_rgb = pointcloud2numpy(pointcloud_camera_frame)
        
        projection_history = list()
        image_y = (numpy_y * self.rs_intrinsics.fy) / numpy_z + self.rs_intrinsics.ppy
        image_x = (numpy_x * self.rs_intrinsics.fx) / numpy_z + self.rs_intrinsics.ppx
        valid_y = np.logical_and(image_y >= 0, image_y < reconstructed_image.shape[0])
        valid_x = np.logical_and(image_x >= 0, image_x < reconstructed_image.shape[1])
        valid_z = np.logical_and(numpy_z > 0, np.isfinite(numpy_z))
        valid_idx = np.logical_and(np.logical_and(valid_y, valid_x), valid_z)
        for i in range(len(valid_idx)):
            if valid_idx[i]:
                reconstructed_image[int(image_y[i]), int(image_x[i]), :] = numpy_rgb[i]
                projection_history.append([image_x[i], image_y[i]])
        return reconstructed_image.astype(np.uint8), projection_history, numpy_x, numpy_y, numpy_z

    def publish_debug_image(self, debug_image, bboxes, save=False):
        if save:
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            image_save_dir = os.path.join(catkin_ws_dir, "data_pose/images/")
            labels_save_dir = os.path.join(catkin_ws_dir, "data_pose/labels/")
            if not os.path.exists(image_save_dir):
                os.makedirs(image_save_dir)
            if not os.path.exists(labels_save_dir):
                os.makedirs(labels_save_dir)
            now = datetime.datetime.now()
            date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(os.path.join(image_save_dir, f"{date_time}.jpg"), cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            with open(os.path.join(labels_save_dir, f"{date_time}.txt"), "w") as file:
                for i, label in enumerate(bboxes):
                    file.write(' '.join(str(e) for e in label))
                    file.write('\n')#New line
        
        self.grasp_candidates_raw_debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8"))
        decoded_debug_image = copy.deepcopy(debug_image)
        color_blue = (0,0,255)
        for i in range(bboxes.shape[0]):
            if i ==1:
                color_blue = (0,255,255)#Draw the most condifent different
            bbox = bboxes[i]
            bbox[[1,2,3,4,5,6,8,9,11,12]] *= debug_image.shape[0]
            x,y,w,h = bbox[1:5]#Bbox
            debug_image = cv2.rectangle(debug_image, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (255,255,0), 2)
            x_c,y_c,x_l,y_l,x_r,y_r = int(bbox[5]), int(bbox[6]), int(bbox[8]), int(bbox[9]), int(bbox[11]), int(bbox[12])
            debug_image = cv2.circle(debug_image, (x_c,y_c), 5, color_blue, -1)
            debug_image = cv2.circle(debug_image, (x_l,y_l), 5, color_blue, -1)
            debug_image = cv2.circle(debug_image, (x_r,y_r), 5, color_blue, -1)
            #Draw center and orientation for decoded image
            p1, p2 = (x_c -(y_r-y_l), y_c +(x_r-x_l)), (x_c +(y_r-y_l), y_c -(x_r-x_l))
            decoded_debug_image = cv2.line(decoded_debug_image, p1, p2, color_blue, 2)
            decoded_debug_image = cv2.circle(decoded_debug_image, (x_c,y_c), 5, color_blue, -1)

        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        ros_decoded_image_msg = self.bridge.cv2_to_imgmsg(decoded_debug_image, encoding="rgb8")
        self.grasp_candidates_debug_pub.publish(ros_image_msg)
        self.grasp_candidates_decoded_debug_pub.publish(ros_decoded_image_msg)
    
    def draw_poses(self, rgb_image):
        annotate_object = DrawKeypoints(rgb_image)
        annotate_object.draw()
        self.bboxes = np.asarray(annotate_object.labels, dtype=np.float64)
        self.save_annotation = annotate_object.save

class DrawKeypoints():
    def __init__(self, image):
        self.lock = Lock()
        self.image, self.image_copy = image, copy.deepcopy(image)
        self.labels = []
        self.save = False
        self._reset()
    
    def _reset(self):
        self.x_c, self.y_c, self.x_l, self.y_l, self.x_r, self.y_r= None, None, None, None, None, None

    def click_event(self, event,x,y,flags,param):
        if event == cv2.EVENT_MBUTTONDOWN and self.x_c is not None and self.x_l is not None and self.x_r is not None:
            #Draw bbox
            diff_x = abs(self.x_r-self.x_l)
            diff_y = abs(self.y_r-self.y_l)
            center = [int(diff_x/2+self.x_l), int(diff_y/2+min(self.y_l, self.y_r))]
            bbox_size = 1.5*max(diff_x, diff_y)
            bbox_tl_x = int(center[0]-bbox_size/2)
            bbox_tl_y = int(center[1]-bbox_size/2)
            bbox_br_x = int(center[0]+bbox_size/2)
            bbox_br_y = int(center[1]+bbox_size/2)
            cv2.rectangle(self.image, (bbox_tl_x, bbox_tl_y), (bbox_br_x,bbox_br_y), (255,255,0), 1)
            
            #Save the label #ALL POINTS ARE NORMALIZED!!!!
            #Class x_tl y_tl x_br y_br x_n y_n v_n -> v: v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible
            img_w = self.image.shape[1]
            img_h = self.image.shape[0]
            self.labels.append([int(0), center[0]/img_w, center[1]/img_h, bbox_size/img_w, bbox_size/img_h, self.x_c/img_w, self.y_c/img_h, 2, self.x_l/img_w, self.y_l/img_h, 2, self.x_r/img_w, self.y_r/img_h, 2])
            
            #Draw angle for visualizing
            p1 = [self.x_c -(self.y_r-self.y_l), self.y_c +(self.x_r-self.x_l)]
            p2 = [self.x_c +(self.y_r-self.y_l), self.y_c -(self.x_r-self.x_l)]
            cv2.line(self.image, p1, p2, (255,0,0), 2)
            self.image_copy = copy.deepcopy(self.image)
            self._reset()
            return
        if event == cv2.EVENT_LBUTTONDOWN and self.x_r is not None:#reset
            self._reset()
            return
    
        if event == cv2.EVENT_LBUTTONDOWN and self.x_l is not None:#Right point
            self.x_r, self.y_r = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)
        
        elif event == cv2.EVENT_LBUTTONDOWN and self.x_c is not None:#Left point
            self.x_l, self.y_l = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)

        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.image_copy is not None:
                self.image = self.image_copy
            self.image_copy = copy.deepcopy(self.image)
            self.x_c, self.y_c = x,y
            cv2.circle(self.image, (x,y), 5, (255,255,255), 1)

    def draw(self): 
        cv2.namedWindow('select_grasp_candidates')
        cv2.setMouseCallback('select_grasp_candidates', self.click_event)
        while(True):
            cv2.imshow('select_grasp_candidates', self.image[...,::-1])
            pressedKey = cv2.waitKey(20) & 0xFF
            if pressedKey == 27:#Close with escape, dont save annotation
                break
            if pressedKey == 13:#Close with enter, save annotation!
                self.save = True
                break
        try:
            cv2.destroyWindow('select_grasp_candidates')
        except cv2.error:
            print("Window already closed. Ignoring")
        cv2.waitKey(100)