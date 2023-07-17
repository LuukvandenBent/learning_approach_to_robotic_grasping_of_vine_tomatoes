import rospy
import os
import sys
import numpy as np
import torch
import rospkg
import tf2_ros
import cv2
from cv_bridge import CvBridge
import datetime
import importlib
from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf2_sensor_msgs
from sensor_msgs.msg import Image, CameraInfo
import copy
import shutil
import pickle

from grasp.srv import choose_grasp_pose_from_candidates_command

from common.util import camera_info2rs_intrinsics, pointcloud2numpy, pointcloud2image, pointcloud2depthimage, reproject_point
from common.transforms import find_transform, transform_pose, transform_pose_array

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))#Needed for import lib

class ChooseGraspPoseFromCandidates():
    def __init__(self, NODE_NAME):
        self.node_name = NODE_NAME
        self.choose_grasp_pose_from_candidates_service = rospy.Service('choose_grasp_pose_from_candidates', choose_grasp_pose_from_candidates_command, self.execute_command)
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.planning_frame = rospy.get_param('/planning_frame')
        self.camera_frame = rospy.get_param('/camera_frame')
        self.rs_intrinsics = None
        self.grasp_pose_debug_pub = rospy.Publisher('grasp_pose_debug', Image, queue_size=1, latch=True)
        self.bridge = CvBridge()
        self.camera_info_sub = rospy.Subscriber("camera/color/camera_info", CameraInfo, self.camera_info_callback)
        
        rospack = rospkg.RosPack()#Data dirs for saving pc data
        grasp_pckg_dir = rospack.get_path('grasp')
        catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
        self.pointcloud_save_dir = os.path.join(catkin_ws_dir, "data_pointnet/pointcloud/")
        
        model = importlib.import_module("pointnet2_cls_ssg")
        checkpoint = torch.load(os.path.join(BASE_DIR, 'weights/pointcloud.pth'), map_location='cpu')
        self.pointcloud_classifier = model.get_model(num_class=2, normal_channel=False)
        self.pointcloud_classifier = self.pointcloud_classifier#.cuda()
        self.pointcloud_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.pointcloud_classifier.eval()
        
        model = importlib.import_module("encoder")
        checkpoint = torch.load(os.path.join(BASE_DIR, 'weights/depth_image_encoder.pth'), map_location='cpu')
        self.depth_image_encoder = model.Encoder(in_channels=1, encoded_space_dim=30, input_size=128)
        self.depth_image_encoder = self.depth_image_encoder#.cuda()
        self.depth_image_encoder.load_state_dict(checkpoint)
        self.depth_image_encoder.eval() 
        
        self.depth_image_svm = pickle.load(open(os.path.join(BASE_DIR, 'weights/depth_image_svm.pickle'), 'rb'))
        self.depth_image_svm = pickle.loads(self.depth_image_svm) 
        
    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)    
        
    def execute_command(self, data):
        print("CHOOSING GRASP POSE FROM CANDIDATES")
        if data.command == "random" or data.command == "center" or data.command == "pointcloud" or data.command == "depth_image":
            self.classifier = data.command
        else:
            return None
                
        try:
            grasp_pose = self.determine_grasp_pose_from_candidates(truss_center=data.truss_center, grasp_candidates=data.grasp_candidates, pointcloud=data.map)
            return grasp_pose
        except Exception as e:
            print(e)
            return None
    
    def determine_grasp_pose_from_candidates(self, truss_center, grasp_candidates, pointcloud):
        number_candidates = len(grasp_candidates.poses)
        print(f"There are a total of {number_candidates} possible grasp candidates!")
        
        pointclouds = list()
        depth_images = list()
        stamped_poses = list()
        grasp_pose = PoseStamped()
        grasp_pose.header.frame_id = grasp_candidates.header.frame_id
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        numpy_x, numpy_y, numpy_z, numpy_rgb = pointcloud2numpy(pointcloud)
        numpy_xyz = np.hstack((numpy_x[...,np.newaxis], numpy_y[...,np.newaxis], numpy_z[...,np.newaxis]))
        for i, pose in enumerate(grasp_candidates.poses):
            grasp_pose.pose = pose
            stamped_poses.append(copy.deepcopy(grasp_pose))
            pc_points, _, depth_image = self.crop_pointcloud_around_grasp(numpy_xyz=np.copy(numpy_xyz), numpy_rgb=np.copy(numpy_rgb), grasp_pose=grasp_pose, save=True, file_name=f"{date_time}_{i}")
            pointclouds.append(pc_points)
            depth_images.append(depth_image/255)

        pred = None
        if self.classifier == "random":
            print("Chosing a random one !!")
            grasp_index = np.random.randint(0, number_candidates)
        elif self.classifier == "center":
            print("Chosing closest to center")
            truss_pos_array = np.array([truss_center.pose.position.x, truss_center.pose.position.y, truss_center.pose.position.z])
            grasp_pos_array = np.zeros((len(stamped_poses),3))
            for i,pose in enumerate(stamped_poses):
                grasp_pos_array[i,:] = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            distances = np.sqrt(np.sum((grasp_pos_array - truss_pos_array)**2, axis=1))
            grasp_index = np.argmin(distances)                
        elif self.classifier == "pointcloud":
            pointclouds_downsampled = list()
            print("Classifying the pointclouds")
            for pc in pointclouds:
                #pointclouds_downsampled.append(self.farthest_point_sample(pc, 2048))
                pointclouds_downsampled.append(self.uniform_point_sample(pc, 2048))
            with torch.no_grad():
                points = torch.Tensor(np.array(pointclouds_downsampled).astype(np.float32))#.cuda()
                points = points.transpose(2, 1)
                pred, _ = self.pointcloud_classifier(points)#(log?) probabilities
                pred = pred.cpu()
                grasp_index = pred[:, 0].argmax().item()#class: success, failure
        elif self.classifier == "depth_image":
            print("Classifying the depth images")
            with torch.no_grad():
                image_batch = torch.Tensor(np.array(depth_images).astype(np.float32))#.cuda()
                image_batch = image_batch.transpose(3,1)
                latent_space = self.depth_image_encoder(image_batch)
                latent_space = latent_space.cpu().detach().numpy()
                pred = self.depth_image_svm.predict_proba(latent_space)
                pred = np.log(pred)#Same as pointcloud format
                pred[:, [0, 1]] = pred[:, [1, 0]]#Swap from failure,success to success,failure
                grasp_index = pred[:, 0].argmax()#class: success, failure
            pass
        else:
            print("No valid classifier, chosing a random one grasp point !!")
            grasp_index = np.random.randint(0, number_candidates)
        
        self.visualize_grasp_candidates(pointcloud=pointcloud, grasp_candidates=stamped_poses, confidence=pred, grasp_index=grasp_index)
        
        self.move_unlabeled_pointclouds(skip=grasp_index)#Move all pointclouds that are not tried to the unlabeled folder
        grasp_pose = stamped_poses[grasp_index]
        return grasp_pose
    
    def crop_pointcloud_around_grasp(self, numpy_xyz, numpy_rgb, grasp_pose=None, save=False, file_name=None):
        if numpy_rgb is None:
            numpy_rgb = np.zeros(np.shape(numpy_xyz))
        #Todo assert pointcloud and grasp pose have same frame_id
        max_distance = 0.05
        if grasp_pose is not None:
            numpy_xyz[:,0] -= grasp_pose.pose.position.x#Center around grasp_pose
            numpy_xyz[:,1] -= grasp_pose.pose.position.y
            numpy_xyz[:,2] -= grasp_pose.pose.position.z
            theta = euler_from_quaternion([grasp_pose.pose.orientation.x, grasp_pose.pose.orientation.y, grasp_pose.pose.orientation.z, grasp_pose.pose.orientation.w])[2]
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],#Rotate around z to align with grasp pose
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
            numpy_xyz = numpy_xyz@rot_matrix#np.dot(rot_matrix, numpy_xyz.T).T
            indexes = np.all(np.abs(numpy_xyz) <= max_distance, axis=1)#Filter on distance
            numpy_xyz, numpy_rgb = numpy_xyz[indexes], numpy_rgb[indexes]
        #numpy_xyz -= np.mean(numpy_xyz, axis=0)#Center around 0#todo not needed!
        numpy_xyz /= max_distance#Normalize

        depth_image = pointcloud2depthimage(copy.deepcopy(numpy_xyz))
        
        if save:
            if not os.path.exists(self.pointcloud_save_dir):
                os.makedirs(self.pointcloud_save_dir)
            xyzrgb = np.hstack((numpy_xyz, numpy_rgb))
            np.savetxt(os.path.join(self.pointcloud_save_dir, f"{file_name}.txt"), xyzrgb, delimiter=',')
        return numpy_xyz, numpy_rgb, depth_image
    
    def farthest_point_sample(self, point, npoint):
        N, D = point.shape
        xyz = point[:,:3]
        centroids = np.zeros((npoint,))
        distance = np.ones((N,)) * 1e10
        farthest = np.random.randint(0, N)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance, -1)
        point = point[centroids.astype(np.int32)]
        return point
    
    def uniform_point_sample(self, point, npoint):#Only works when the points are 'ordered'
        indices = np.linspace(0, point.shape[0]-1, npoint, dtype=int)
        return point[indices]
    
    def visualize_grasp_candidates(self, pointcloud, grasp_candidates, confidence, grasp_index):
        transform = find_transform(pointcloud.header.frame_id, self.camera_frame, self.tfBuffer)
        pointcloud_camera_frame = tf2_sensor_msgs.do_transform_cloud(pointcloud, transform)
        rgb_image, _, _, _, _= pointcloud2image(pointcloud_camera_frame, self.rs_intrinsics)
        
        for i, grasp_candidate in enumerate(grasp_candidates):
            grasp_candidate = transform_pose(grasp_candidate, self.camera_frame, self.tfBuffer)
            image_x, image_y = reproject_point(grasp_candidate.pose.position.x, grasp_candidate.pose.position.y, grasp_candidate.pose.position.z, self.rs_intrinsics)
            rgb_image = cv2.circle(rgb_image, (image_x, image_y), 5, (0,0,255), -1)
            if i == grasp_index:#color differently
                rgb_image = cv2.circle(rgb_image, (image_x, image_y), 5, (255,255,0), -1)
            if confidence is not None:
                rgb_image = cv2.putText(rgb_image, str(np.around(np.e**confidence[i, 0].item(), 2)), (image_x, image_y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)
            #todo also draw the angle?
        ros_image_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding="rgb8")
        self.grasp_pose_debug_pub.publish(ros_image_msg)
    
    def move_unlabeled_pointclouds(self, skip):
        pointcloud_unlabeled_dir = os.path.join(self.pointcloud_save_dir, "unlabeled")
        if not os.path.exists(pointcloud_unlabeled_dir):
            os.makedirs(pointcloud_unlabeled_dir)
        for file in os.listdir(self.pointcloud_save_dir):
            if file.endswith(".txt") and not file.endswith(f"{skip}.txt"):
                shutil.move(os.path.join(self.pointcloud_save_dir, file), os.path.join(pointcloud_unlabeled_dir, file))