

from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf
import tf2_ros

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

import numpy as np
from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL


def all_close(goal, actual, tolerance):
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

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True


class MoveGroupPythonInterfaceTutorial(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(MoveGroupPythonInterfaceTutorial, self).__init__()


        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("move_group_python_interface_tutorial", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
	
        
        ## This interface can be used to plan and execute motions:
        group_name = "panda_manipulator" ## or "panda_arm"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        

        self.box_name = ""
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names
        self.listener = tf.TransformListener()
        ###################### TODO：switch between simulation and real robot
        self.true_eef = "/panda_hand_tcp" ## real
        #self.true_eef = '/panda_hand' ## simulation
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        
        # return self.R
    def rotation_matrix_to_quaternion(self,R):
        pass
    def go_to_obs_joint_state(self):

        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 10
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 5.2  
        joint_goal[6] = tau / 8
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    def go_to_joint_state(self):
        
        move_group = self.move_group

       
        ## Planning to a Joint Goal, get the joint values from the group and change some of the values:
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        print("============ current joint values: %s" % joint_goal)
        
        ####################################################################
        ########## TODO: for group_name = panda_manipulator
        #joint_goal[7] = 1
        #joint_goal[8] = 1
        #########################################
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 6  
        joint_goal[6] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()


        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    
    
    def go_to_pose_goal(self,x=0.76,y=-0.25,z=0.2):
        
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
        ## plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x  ### 0.74 - 0.32
        pose_goal.position.y = y
        pose_goal.position.z = z
        
        

        move_group.set_pose_target(pose_goal)

        ## call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        
        move_group.clear_pose_targets()


        # For testing:
        
        current_pose = self.move_group.get_current_pose().pose
        current_joints = move_group.get_current_joint_values()
        print('current_pose',current_pose)
        print('current_joints: ',current_joints)
        return all_close(pose_goal, current_pose, 0.01)
    
    

def main():
    try:
        print("")
        print("----------------------------------------------------------")
        print("Press Ctrl-D to exit at any time")
        print("")
        print(
            "...setting up the moveit_commander ..."
        )
        tutorial = MoveGroupPythonInterfaceTutorial()
        input("============ Press `Enter` to execute a movement using a pose goal ...")
        tutorial.go_to_obs_joint_state()
        input(
            "============ Press `Enter` to execute a movement using a joint state goal ..."
        )
        tutorial.go_to_pose_goal(x=0.76,y=-0.20,z=0.18)
       
        
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()

