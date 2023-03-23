from moveit_commander.conversions import pose_to_list
from geometry_msgs.msg import Pose, PoseStamped
import moveit_msgs.msg
import numpy as np
from shape_msgs.msg import SolidPrimitive    
from tf.transformations import quaternion_from_euler
def all_close(goal, actual, tolerance, only_check_pos=False, only_check_angle=False):#From moveit_tutorials move_group_python_interface
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = np.linalg.norm(np.array([x1, y1, z1])- np.array([x0, y0, z0]))
        # phi = angle between orientations
        cos_phi_half = np.fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        if only_check_pos:
            return d <= tolerance
        elif only_check_angle:
            return cos_phi_half >= np.cos(tolerance / 2.0)
        else:
            return d <= tolerance and cos_phi_half >= np.cos(tolerance / 2.0)

    return False

def create_collision_object(robot, id, dimensions, pose, orientation=[0,0,0]):
    collision_object = moveit_msgs.msg.CollisionObject()
    collision_object.id = id
    collision_object.header.frame_id = robot.get_planning_frame()

    solid = SolidPrimitive()
    solid.type = solid.BOX
    solid.dimensions = dimensions
    collision_object.primitives = [solid]

    object_pose = Pose()
    object_pose.position.x = pose[0]
    object_pose.position.y = pose[1]
    object_pose.position.z = pose[2]
    
    quaternion = quaternion_from_euler(orientation[0], orientation[1], orientation[2])#RPY
    object_pose.orientation.x = quaternion[0]
    object_pose.orientation.y = quaternion[1]
    object_pose.orientation.z = quaternion[2]
    object_pose.orientation.w = quaternion[3]

    collision_object.primitive_poses = [object_pose]
    collision_object.operation = collision_object.ADD
    return collision_object