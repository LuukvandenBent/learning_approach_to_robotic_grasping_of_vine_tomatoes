import rospy
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from sensor_msgs.msg import Image, CameraInfo
import tf2_geometry_msgs
import copy
    
def find_transform(source, target, tfBuffer):
    try:
        transform = tfBuffer.lookup_transform(target,
                                    source,
                                    rospy.Time(0),
                                    rospy.Duration(3.0))
        return transform
    except Exception as e:
        print(e)
        return None
    
def transform_pose_array(data, frame, tfBuffer):
    transform = find_transform(data.header.frame_id, frame, tfBuffer)
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

def transform_pose(data, frame, tfBuffer):
    transform = find_transform(data.header.frame_id, frame, tfBuffer)
    if transform == None:
        return None
    else:
        transformed_data = tf2_geometry_msgs.do_transform_pose(data, transform)
        return transformed_data