#!/usr/bin/env python

from actionlib import SimpleActionServer
from moveit_commander.conversions import pose_to_list
import numpy as np
import rospy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped

from panda_grasp_demo.msg import GraspAction, GraspResult
from panda_grasp_demo.panda_commander import PandaCommander


def multiply_transforms(A, B):
    trans1, rot1 = A[:3], Rotation.from_quat(A[3:])
    trans2, rot2 = B[:3], Rotation.from_quat(B[3:])
    return np.r_[rot1.apply(trans2) + trans1, (rot1 * rot2).as_quat()]


class GraspExecutionNode(object):


    def __init__(self):
        self.panda_commander = PandaCommander("panda_arm")

        self.selected_grasp_pub = rospy.Publisher('/pre_grasp_pose', PoseStamped, queue_size=10)

        self._as = SimpleActionServer("grasp_action", GraspAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()

        rospy.loginfo("Grasp action server ready")

    def execute_cb(self, goal_msg):
        rospy.loginfo("Received grasp pose")

        self.panda_commander.move_group.set_max_velocity_scaling_factor(0.05)
        self.panda_commander.move_group.set_max_acceleration_scaling_factor(0.05)

        grasp_pose_msg = goal_msg.target_grasp_pose.pose
        T_base_grasp = pose_to_list(grasp_pose_msg)

        T_grasp_pregrasp = np.r_[0.0, 0.0, -0.1, 0.0, 0.0, 0.0, 1.0]
        T_base_pregrasp = multiply_transforms(T_base_grasp, T_grasp_pregrasp)

        self.panda_commander.move_gripper(width=0.10)

        rospy.loginfo("Moving to pregrasp pose")
        self.panda_commander.goto_pose_target(T_base_pregrasp.tolist())
        
        rospy.loginfo("Moving to grasp pose")
        self.panda_commander.follow_cartesian_waypoints([T_base_grasp])
        
        self.panda_commander.grasp(0.05)
        
        rospy.loginfo("Retrieving object")
        self.panda_commander.follow_cartesian_waypoints([T_base_pregrasp.tolist()])

        self._as.set_succeeded(GraspResult())


if __name__ == '__main__':
    try:
        rospy.init_node("grasp_execution_node")
        GraspExecutionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
