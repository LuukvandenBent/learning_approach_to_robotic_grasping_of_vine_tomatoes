import rospy
import os
import numpy as np
import torch
import cv2
from cv_bridge import CvBridge
import tf2_ros
import rospkg
import copy
import tf2_geometry_msgs
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from grasp.srv import detect_truss_command, detect_truss_commandResponse
from common.util import camera_info2rs_intrinsics, DepthImageFilter
from common.transforms import transform_pose_array

from threading import Lock
import datetime

class DetectTrussManual():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.image = None
        self.depth_image = None
        self.camera_info = None
        self.rs_intrinsics = None
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.image_callback)
        self.depth_image_sub = rospy.Subscriber("camera/depth/image_rect_raw", Image, self.depth_image_callback)
        self.detect_truss_service = rospy.Service('detect_truss_manual', detect_truss_command, self.execute_command)
        
        self.truss_detection_pub = rospy.Publisher('truss_detection', Image, queue_size=1, latch=True)
        self.truss_detection_raw_pub = rospy.Publisher('truss_detection_raw', Image, queue_size=1, latch=True)
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')

        self.collect_image = False
        self.collect_depth_image = False
        self.draw = False
        self.lock = Lock()

    def image_callback(self, image):
        if self.collect_image:
            self.image = image
            self.collect_image = False
        else:
            pass
    
    def depth_image_callback(self, depth_image):
        if self.collect_depth_image:
            self.depth_image = depth_image
            self.collect_depth_image = False
        else:
            pass
    
    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.rs_intrinsics = camera_info2rs_intrinsics(self.camera_info)
    
    def execute_command(self, command):
        print("SERVICE ARIVED WITH COMMAND: ", command)
        if command.command == "detect_truss_manual":
            self.image, self.depth_image = None, None
            self.collect_image, self.collect_depth_image = True, True
            self.bboxes = None
            counter = 0
            while self.collect_image or self.collect_depth_image:#Save both rgb and depth 
                rospy.sleep(0.1)
                counter += 1
                if counter > 10:
                    print("No images comming in")
                    return None
            self.bridge = CvBridge()
            truss_data = self.detect_truss(image=self.image, depth_image=self.depth_image)
            return truss_data
        else:
            return None

    def detect_truss(self, image=None, depth_image=None):
        rgb_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_image)
        #rescale
        image_size = 640
        pad_size = int((rgb_image.shape[1]-rgb_image.shape[0])/2)
        preprocessed_image = cv2.copyMakeBorder(rgb_image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, None, 0)
        image_scale = preprocessed_image.shape[0]/image_size
        preprocessed_image = cv2.resize(preprocessed_image, (image_size,image_size), interpolation = cv2.INTER_AREA)
        self.preprocessed_image = preprocessed_image#Used for running in main thread
        #Run model
        with self.lock:
            self.draw = True
        
        with self.lock:#Do the drawing in the main thread
            while self.bboxes is None:
                rospy.sleep(1)
        bboxes = copy.deepcopy(self.bboxes)
        save_annotation = copy.deepcopy(self.save_annotation)
        self.bboxes, self.save_annotation = None, None
        
        self.publish_debug_image(debug_image=copy.deepcopy(preprocessed_image), bboxes=bboxes, save=save_annotation)
        if len(bboxes) == 0:#No detections
            return None
        
        bbox = bboxes[0]#Invert the rescaling
        bbox[:8] *= image_scale#Dont scale the condifence/class
        bbox[1:8:2] -= pad_size#Move the y coordinates to account for padding
        truss_data = self.generate_truss_data(image=image, depth_image=depth_image, bbox=bbox)
        
        return truss_data
    
    def generate_truss_data(self, image, depth_image, bbox):
        mid_point = (int((bbox[0]+bbox[4])/2), int((bbox[1]+bbox[5])/2))
        if np.linalg.norm(bbox[0:2]-bbox[2:4]) < np.linalg.norm(bbox[0:2]-bbox[6:8]):#Get the correct angle
            bbox_angle = np.arctan2(bbox[1]-bbox[3], bbox[0]-bbox[2])
        else:
            bbox_angle = np.arctan2(bbox[1]-bbox[7], bbox[0]-bbox[6])
        depth_image_filter = DepthImageFilter(depth_image, self.rs_intrinsics, patch_size=30)
        depth = depth_image_filter.get_depth(mid_point[1], mid_point[0])#Assume same depth for the corners
        depth = depth/1000#Convert from mm to m
        xyz_mid_point = depth_image_filter.deproject(mid_point[1], mid_point[0], depth=depth)
        
        truss_data = PoseArray()
        truss_data.header.frame_id = image.header.frame_id
        truss_center, truss_edge1, truss_edge2, truss_edge3, truss_edge4  = Pose(), Pose(), Pose(), Pose(), Pose()
        
        truss_center.position.x, truss_center.position.y, truss_center.position.z = xyz_mid_point[0], xyz_mid_point[1], xyz_mid_point[2]
        truss_center.orientation.x, truss_center.orientation.y, truss_center.orientation.z, truss_center.orientation.w = quaternion_from_euler(0, 0, bbox_angle)

        for i, truss in enumerate([truss_edge1, truss_edge2, truss_edge3, truss_edge4]):#4 corners
            xyz = depth_image_filter.deproject(bbox[2*i+1], bbox[2*i], depth=depth)
            truss.position.x, truss.position.y, truss.position.z = xyz[0], xyz[1], xyz[2]
        
        truss_data.poses.extend([truss_center, truss_edge1, truss_edge2, truss_edge3, truss_edge4])
        truss_data = transform_pose_array(truss_data, self.planning_frame, self.tfBuffer)
        
        _,_, z_angle = euler_from_quaternion([truss_data.poses[0].orientation.x, truss_data.poses[0].orientation.y, truss_data.poses[0].orientation.z, truss_data.poses[0].orientation.w])#
        if z_angle < -np.pi/2:#Add/subtract 180deg since gripper is symmetrical, to avoid hitting joint limits
            z_angle += np.pi
        elif z_angle > np.pi/2:
            z_angle -= np.pi
        truss_data.poses[0].orientation.x, truss_data.poses[0].orientation.y, truss_data.poses[0].orientation.z, truss_data.poses[0].orientation.w  = quaternion_from_euler(-np.pi, np.pi/2, z_angle)#Orientation with respect to fake_camera_bottom_screw RPY: -pi, pi/2, angle

        print("Truss found at :", truss_data.poses[0].position)
        return truss_data
    
    def publish_debug_image(self, debug_image, bboxes, save=False):
        if save:
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            image_save_dir = os.path.join(catkin_ws_dir, "data_obb/images/")
            labels_save_dir = os.path.join(catkin_ws_dir, "data_obb/labels/")
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
                    file.write(' ')
                    file.write(' '.join(str(e) for e in ["vine-tomatoes", 0]))
                    file.write('\n')#New line
        
        self.truss_detection_raw_pub.publish(self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8"))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            edge_coordinates = list()
            for j in range(4):
                edge_coordinates.append((int(bbox[j*2]), int(bbox[j*2+1]))) 
            debug_image = cv2.line(debug_image, edge_coordinates[0], edge_coordinates[1], (255,255,0), 2)
            debug_image = cv2.line(debug_image, edge_coordinates[1], edge_coordinates[2], (255,255,0), 2)
            debug_image = cv2.line(debug_image, edge_coordinates[2], edge_coordinates[3], (255,255,0), 2)
            debug_image = cv2.line(debug_image, edge_coordinates[3], edge_coordinates[0], (255,255,0), 2)
            debug_image = cv2.putText(debug_image, str(i), edge_coordinates[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)


        ros_image_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="rgb8")
        self.truss_detection_pub.publish(ros_image_msg)
    
    def draw_bboxes(self, rgb_image):
        annotate_object = DrawBboxes()
        annotate_object.reset(copy.deepcopy(rgb_image))
        annotate_object.draw()
        self.bboxes = np.asarray(annotate_object.bboxes, dtype=np.float64)
        self.save_annotation = annotate_object.save

class DrawBboxes():
    def __init__(self):
        self.lock = Lock()
        self.reset()
    
    def reset(self, image=None):
        self.image = image
        self.image_copy = copy.deepcopy(self.image)
        self.save = False
        self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = None, None, None, None, None, None, None, None
        self.bboxes = list()

    def click_event(self, event,x,y,flags,param):
        if event == cv2.EVENT_MBUTTONDOWN and self.x4 is not None:
            #Draw bbox
            box = cv2.minAreaRect(np.array([[self.x1, self.y1], [self.x2, self.y2], [self.x3, self.y3], [self.x4, self.y4]]))
            points = np.int0(cv2.boxPoints(box))
            
            min_index = np.argmin(points[:,0])#Put the found list in the same ordering format as yolov5_OBB
            points = np.roll(points, shift=-min_index, axis=0)
            
            cv2.line(self.image, points[0], points[1], (255,255,0), 5)
            cv2.line(self.image, points[1], points[2], (255,255,0), 5)
            cv2.line(self.image, points[2], points[3], (255,255,0), 5)
            cv2.line(self.image, points[3], points[0], (255,255,0), 5)
            self.image_copy = copy.deepcopy(self.image)
            self.bboxes.append([points[0][0], points[0][1], points[1][0], points[1][1], points[2][0], points[2][1], points[3][0], points[3][1]])
        
        elif event == cv2.EVENT_LBUTTONDOWN and self.x4 is not None:
            self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4 = None, None, None, None, None, None, None, None
            self.image = copy.deepcopy(self.image_copy)
        elif event == cv2.EVENT_LBUTTONDOWN and self.x3 is not None:
            self.x4,self.y4= int(x), int(y)
            self.image = cv2.circle(self.image, (int(x), int(y)), 5, (255,255,0), -1)
        elif event == cv2.EVENT_LBUTTONDOWN and self.x2 is not None:
            self.x3,self.y3= int(x), int(y)
            self.image = cv2.circle(self.image, (int(x), int(y)), 5, (255,255,0), -1)
        elif event == cv2.EVENT_LBUTTONDOWN and self.x1 is not None:
            self.x2,self.y2= int(x), int(y)
            self.image = cv2.circle(self.image, (int(x), int(y)), 5, (255,255,0), -1)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.x1,self.y1= int(x), int(y)
            self.image = cv2.circle(self.image, (int(x), int(y)), 5, (255,255,0), -1)

    def draw(self): 
        cv2.namedWindow('create_bboxes')
        cv2.setMouseCallback('create_bboxes', self.click_event)
        while(True):
            cv2.imshow('create_bboxes', self.image[...,::-1])
            pressedKey = cv2.waitKey(20) & 0xFF
            if pressedKey == 27:#Close with escape, dont save annotation
                break
            if pressedKey == 13:#Close with enter, save annotation!
                self.save = True
                break
        try:
            cv2.destroyWindow('create_bboxes')
        except cv2.error:
            print("Window already closed. Ignoring")
        cv2.waitKey(100)