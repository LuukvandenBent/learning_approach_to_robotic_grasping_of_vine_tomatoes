import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import ctypes
import struct
import sensor_msgs.point_cloud2
import copy
from tf.transformations import quaternion_from_euler, euler_from_quaternion

def camera_info2rs_intrinsics(camera_info):

    # init object
    rs_intrinsics = rs.intrinsics()

    # dimensions
    rs_intrinsics.width = camera_info.width
    rs_intrinsics.height = camera_info.height

    # principal point coordinates
    rs_intrinsics.ppx = camera_info.K[2]
    rs_intrinsics.ppy = camera_info.K[5]

    # focal point
    rs_intrinsics.fx = camera_info.K[0]
    rs_intrinsics.fy = camera_info.K[4]

    return rs_intrinsics
  
class DepthImageFilter(object):
    """DepthImageFilter"""

    def __init__(self, image, intrinsics, patch_size=5):
        self.image = image
        self.intrinsics = intrinsics
        self.patch_radius = int((patch_size - 1) / 2)

        # patch
        self.min_row = 0
        self.max_row = self.image.shape[0]
        self.min_col = 0
        self.max_col = self.image.shape[1]

    def generate_patch(self, row, col):
        """Returns a patch which can be used for filtering an image"""
        row = int(round(row))
        col = int(round(col))

        row_start = max([row - self.patch_radius, self.min_row])
        row_end = min([row + self.patch_radius, self.max_row - 1]) + 1

        col_start = max([col - self.patch_radius, self.min_col])
        col_end = min([col + self.patch_radius, self.max_col - 1]) + 1

        rows = np.arange(row_start, row_end)
        cols = np.arange(col_start, col_end)
        return rows, cols

    def deproject(self, row, col, depth=None, segment=None):
        """
        Deproject a 2D coordinate to a 3D point using the depth image, if an depth is provided the depth image is
        ignored.
        """

        if depth is None:
            depth = self.get_depth(row, col, segment=segment)

        if np.isnan(depth):
            return 3 * [np.nan]

        point = rs.rs2_deproject_pixel_to_point(self.intrinsics, [col, row], depth)
        return point
      
    def get_depth(self, row, col, segment=None):
      """get the filtered depth from an depth image at location row, col"""
      rows, cols = self.generate_patch(row, col)

      if segment is None:
          depth_patch = self.image[rows[:, np.newaxis], cols]
      else:
          depth_patch = self.image[segment > 0]

      non_zero = np.nonzero(depth_patch)
      depth_patch_non_zero = depth_patch[non_zero]
      #return np.median(depth_patch_non_zero)
      return np.percentile(depth_patch_non_zero, 10)

def pointcloud2numpy(pointcloud, rgb=True):
    numpy_x = np.array(list(sensor_msgs.point_cloud2.read_points(pointcloud, skip_nans=True, field_names=['x']))).squeeze()
    numpy_y = np.array(list(sensor_msgs.point_cloud2.read_points(pointcloud, skip_nans=True, field_names=['y']))).squeeze()
    numpy_z = np.array(list(sensor_msgs.point_cloud2.read_points(pointcloud, skip_nans=True, field_names=['z']))).squeeze()
    numpy_rgb = np.array(list(sensor_msgs.point_cloud2.read_points(pointcloud, skip_nans=True, field_names=['rgb'])))
    if rgb:
        rgb_list = []
        for rgb in numpy_rgb:
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,rgb)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            rgb_list.append([r, g, b])
        numpy_rgb = np.asarray(rgb_list).astype('int')
    else:
        numpy_rgb = None
    return numpy_x, numpy_y, numpy_z, numpy_rgb

def pointcloud2image(pointcloud, rs_intrinsics):
    reconstructed_image = np.zeros((rs_intrinsics.height, rs_intrinsics.width, 3))
    numpy_x, numpy_y, numpy_z, numpy_rgb = pointcloud2numpy(pointcloud)
    
    projection_history = list()
    image_y = (numpy_y * rs_intrinsics.fy) / numpy_z + rs_intrinsics.ppy
    image_x = (numpy_x * rs_intrinsics.fx) / numpy_z + rs_intrinsics.ppx
    valid_y = np.logical_and(image_y >= 0, image_y < reconstructed_image.shape[0])
    valid_x = np.logical_and(image_x >= 0, image_x < reconstructed_image.shape[1])
    valid_z = np.logical_and(numpy_z > 0, np.isfinite(numpy_z))
    valid_idx = np.logical_and(np.logical_and(valid_y, valid_x), valid_z)
    for i in range(len(valid_idx)):
        if valid_idx[i]:
            reconstructed_image[int(image_y[i]), int(image_x[i]), :] = numpy_rgb[i]
            projection_history.append([image_x[i], image_y[i]])
    return reconstructed_image.astype(np.uint8), projection_history, numpy_x, numpy_y, numpy_z

def pointcloud2depthimage(numpy_xyz):#todo make input pointcloud2 #todo make generic func
    image_size = 128
    #normalize to [0,1] and scale according to required image size
    numpy_x, numpy_y, numpy_z = (numpy_xyz[:,0]+1)/2*image_size, (numpy_xyz[:,1]+1)/2*image_size, (numpy_xyz[:,2]+1)/2*image_size
    
    image_y = numpy_y
    image_x = numpy_x
    reconstructed_image = np.zeros((int(image_size), int(image_size), 1))
    valid_y = np.logical_and(image_y >= 0, image_y < reconstructed_image.shape[0])
    valid_x = np.logical_and(image_x >= 0, image_x < reconstructed_image.shape[1])
    valid_z = np.isfinite(numpy_z)
    valid_idx = np.logical_and(np.logical_and(valid_y, valid_x), valid_z)
    
    for i in range(len(valid_idx)):
        if valid_idx[i]:
            reconstructed_image[int(image_y[i]), int(image_x[i]), :] = numpy_z[i]
    min_val = np.min(reconstructed_image[np.nonzero(reconstructed_image)])
    max_val = np.max(reconstructed_image[np.nonzero(reconstructed_image)])
    reconstructed_image[np.nonzero(reconstructed_image)] = ((reconstructed_image[np.nonzero(reconstructed_image)] - min_val) / (max_val - min_val)) * 200+55
    reconstructed_image = reconstructed_image.astype(np.uint8)
    return reconstructed_image

def reproject_point(x, y, z, rs_intrinsics):
    image_y = (y * rs_intrinsics.fy) / z + rs_intrinsics.ppy
    image_x = (x * rs_intrinsics.fx) / z + rs_intrinsics.ppx
    return int(image_x), int(image_y)
    
def flip_z_rotation_stamped_pose(pose):
    flipped_goal_pose = copy.deepcopy(pose)
    angles = list(euler_from_quaternion((flipped_goal_pose.pose.orientation.x, flipped_goal_pose.pose.orientation.y, flipped_goal_pose.pose.orientation.z, flipped_goal_pose.pose.orientation.w)))
    angles[2] += np.pi
    quaternion = quaternion_from_euler(angles[0], angles[1], angles[2])
    flipped_goal_pose.pose.orientation.x, flipped_goal_pose.pose.orientation.y , flipped_goal_pose.pose.orientation.z , flipped_goal_pose.pose.orientation.w = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return flipped_goal_pose