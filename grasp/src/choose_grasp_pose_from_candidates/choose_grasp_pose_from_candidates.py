import rospy
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
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
import threading
from sklearn.neighbors import KNeighborsClassifier

from grasp.srv import choose_grasp_pose_from_candidates_command

from load_data import CustomDataset

from common.util import camera_info2rs_intrinsics, pointcloud2numpy, pointcloud2image, pointcloud2depthimage, reproject_point
from common.transforms import find_transform, transform_pose, transform_pose_array
from common.download_model import download_from_google_drive

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
        self.pointcloud_save_dir = os.path.join(catkin_ws_dir, "experiments/pointcloud/")
        
        file_id = "1QSNF9zeecNudcsDd9i5JPskQDEeWv9TY"  # Replace this with the Google Drive file ID
        download_from_google_drive(file_id, os.path.join(BASE_DIR, 'weights/depth_image_encoder.pth'))
        model = importlib.import_module("encoder")
        checkpoint = torch.load(os.path.join(BASE_DIR, 'weights/depth_image_encoder.pth'), map_location='cpu')
        self.encoded_space_dim=30
        self.input_size = 128
        self.depth_image_encoder = model.Encoder(in_channels=1, encoded_space_dim=self.encoded_space_dim, input_size=self.input_size)
        self.depth_image_encoder = self.depth_image_encoder#.cuda()
        self.depth_image_encoder.load_state_dict(checkpoint)
        self.depth_image_encoder.eval()
        
        self.stored_encoded_data, self.stored_encoded_labels = self.get_data(encoder=self.depth_image_encoder, location=os.path.join(BASE_DIR, 'stored_data/'))
        
    def camera_info_callback(self, msg):
        if self.rs_intrinsics is None:
            self.rs_intrinsics = camera_info2rs_intrinsics(msg)    
        
    def execute_command(self, data):
        print("CHOOSING GRASP POSE FROM CANDIDATES")
        if data.command == "random" or data.command == "center" or data.command == "depth_image":
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

        #for saving distance to center
        truss_pos_array = np.array([truss_center.pose.position.x, truss_center.pose.position.y, truss_center.pose.position.z])
        grasp_pos_array = np.zeros((len(stamped_poses),3))
        for i,pose in enumerate(stamped_poses):
            grasp_pos_array[i,:] = [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
        distances = np.sqrt(np.sum((grasp_pos_array - truss_pos_array)**2, axis=1))
        if not os.path.exists(self.pointcloud_save_dir):
            os.makedirs(self.pointcloud_save_dir)
        np.save(os.path.join(self.pointcloud_save_dir, "distances.npy"), distances)
        #
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
        elif self.classifier == "depth_image":
            print("Classifying the depth images")
            depth_image_knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
            
            rospack = rospkg.RosPack()
            grasp_pckg_dir = rospack.get_path('grasp')
            catkin_ws_dir = os.path.dirname(os.path.dirname(os.path.dirname(grasp_pckg_dir)))
            saved_exp_dir = os.path.join(catkin_ws_dir, "saved_experiments")
            length_stored_encoded_labels = len(self.stored_encoded_labels) if self.stored_encoded_labels is not None else 0
            print("Found ", length_stored_encoded_labels, "stored samples")
            online_encoded_data, online_encoded_labels = self.get_data(encoder=self.depth_image_encoder, location=saved_exp_dir, online=True)
            #online_encoded_data, online_encoded_labels = None, None #todo ADD NOW FOR EXPERIMENT
            length_online_encoded_labels = len(online_encoded_labels) if online_encoded_labels is not None else 0
            print("Found ", length_online_encoded_labels, "online samples")
            if online_encoded_data is not None:
                data_all = np.vstack([self.stored_encoded_data, online_encoded_data])
                labels_all = np.vstack([self.stored_encoded_labels, online_encoded_labels])
            elif self.stored_encoded_data is not None:
                data_all = self.stored_encoded_data
                labels_all = self.stored_encoded_labels
            else:
                data_all = None
                labels_all = None
            if data_all is not None:
                depth_image_knn.fit(data_all, labels_all.squeeze())
                print("KNN FITTED")
                with torch.no_grad():
                    augmented_depth_images = []
                    augmentations = ['000','001','010','011','100','101','110','111']
                    for depth_image in depth_images:
                        for augment in augmentations:
                            if int(augment[2]) == 1:
                                depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)
                            if int(augment[1]) == 1:
                                depth_image = cv2.flip(depth_image, 0)#vertical
                            if int(augment[0]) == 1:
                                depth_image = cv2.flip(depth_image, 1)#horizontal
                            augmented_depth_images.append(depth_image)
                    number_of_augmentations = len(augmentations)
                    image_batch = torch.Tensor(np.array(augmented_depth_images)[:, np.newaxis].astype(np.float32))#.cuda()
                    latent_space = self.depth_image_encoder(image_batch)
                    latent_space = latent_space.cpu().detach().numpy()
                    distances_extended = np.repeat(distances, number_of_augmentations)
                    pred = depth_image_knn.predict_proba(np.hstack([latent_space, distances_extended[:, np.newaxis]]))
                    pred = pred.reshape(-1, number_of_augmentations, pred.shape[1])
                    pred = np.mean(pred, axis=1)
                    pred += 1e-10#Safety for log
                    pred = np.log(pred)#Same as pointcloud format
                    pred[:, [0, 1]] = pred[:, [1, 0]]#Swap from failure,success to success,failure
                    grasp_index = pred[:, 0].argmax()#class: success, failure
            else:
                print("no stored or online data to compute, chosing random grasp point")
                grasp_index = np.random.randint(0, number_candidates) 
        else:
            print("No valid classifier, chosing a random one grasp point !!")
            grasp_index = np.random.randint(0, number_candidates)
        
        self.visualize_grasp_candidates(pointcloud=pointcloud, grasp_candidates=stamped_poses, confidence=pred, grasp_index=grasp_index)
        
        thread = threading.Thread(target=self.move_unlabeled_pointclouds(skip=grasp_index))#Move all pointclouds that are not tried to the unlabeled folder
        thread.start()
        
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
            numpy_xyz = numpy_xyz@rot_matrix
            indexes = np.all(np.abs(numpy_xyz) <= max_distance, axis=1)#Filter on distance
            numpy_xyz, numpy_rgb = numpy_xyz[indexes], numpy_rgb[indexes]
        numpy_xyz /= max_distance#Normalize
        depth_image = pointcloud2depthimage(copy.deepcopy(numpy_xyz))
        depth_image = self.fix_depth_image(depth_image)
        
        if save:
            thread = threading.Thread(target=self.save_pointcloud(numpy_xyz, numpy_rgb, file_name, depth_image=depth_image))
            thread.start()
            
        return numpy_xyz, numpy_rgb, depth_image
    
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
            elif file.endswith(".png") and not file.endswith(f"{skip}.png"):
                shutil.move(os.path.join(self.pointcloud_save_dir, file), os.path.join(pointcloud_unlabeled_dir, file))
    
    def save_pointcloud(self, numpy_xyz, numpy_rgb, file_name, depth_image=None):
        if not os.path.exists(self.pointcloud_save_dir):
                os.makedirs(self.pointcloud_save_dir)
        xyzrgb = np.hstack((numpy_xyz, numpy_rgb))
        np.savetxt(os.path.join(self.pointcloud_save_dir, f"{file_name}.txt"), xyzrgb, delimiter=',')
        if depth_image is not None:
            cv2.imwrite(os.path.join(self.pointcloud_save_dir, f"{file_name}.png"), depth_image)
    
    def get_data(self, encoder, location, online=False):
        labeled_dataset = CustomDataset(path=location, size=0.02, input_size=self.input_size, online=online)
        if len(labeled_dataset) == 0:
            return None, None
        labeled_loader = DataLoader(dataset=labeled_dataset, batch_size=512, shuffle=True)
        latent_space_all = np.array([]).reshape(0,self.encoded_space_dim)
        labels_all = np.array([]).reshape(0,1)
        distance_all = np.array([]).reshape(0,1)
        for image_batch, labels, distance in labeled_loader:
            encoded_data = encoder(image_batch)
            latent_space = encoded_data.cpu().detach().numpy()
            latent_space_all = np.vstack([latent_space_all,latent_space])
            labels = labels.cpu().detach().numpy()
            labels_all = np.vstack([labels_all,labels])
            distance = distance.numpy()
            distance_all = np.vstack([distance_all,distance])
        stored_encoded_data = np.hstack([latent_space_all, distance_all])
        stored_encoded_labels = labels_all
        return stored_encoded_data, stored_encoded_labels
    
    def fix_depth_image(self, img):#todo not hardcode the numbers
        old_size = 0.05#captured size
        new_size = [0.02, 0.02]#y,x
        img_size = img.shape[0]
        img = img[int(img_size//2-new_size[0]/old_size/2*img_size):int(img_size//2+new_size[0]/old_size/2*img_size), int(img_size//2-new_size[1]/old_size/2*img_size):int(img_size//2+new_size[1]/old_size/2*img_size)]
        img = cv2.resize(img, (128, 128), interpolation = cv2.INTER_AREA)

        normalize_value = np.max(img)
        nonzero_rows, nonzero_cols = np.where(img != 0)
        img[nonzero_rows, nonzero_cols] = cv2.add(img[nonzero_rows, nonzero_cols], int(255-normalize_value)).squeeze()
        return img
