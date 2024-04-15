#!/usr/bin/env python
from __future__ import print_function
from six.moves import input
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tf
import numpy as np
import open3d as o3d
import cv2
import torch
import sys
import copy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf2_ros
import pickle
from geometry_msgs.msg import PoseStamped
from moma_utils.ros.panda import PandaGripperClient
from stable_baselines3 import PPO
path_save = './data/'
try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

## END_SUB_TUTORIAL

def get_x_y(tsdf,occupancy):
    obs = np.zeros([1,50,50,2])
    obs[0,:,:,0] = occupancy.copy()
    obs[0,:,:,1] = tsdf.copy()
    checkpoint = './data/model.zip'
    agent = PPO.load(checkpoint)
    actions, _ = agent.predict(obs, deterministic=True)
    obs_tensor = torch.from_numpy(obs)
    # print(obs_tensor.size())
    obs_tensor = obs_tensor.permute(0,3,1,2)
    # print(obs_tensor.size())
    actions_tensor_tmp =  torch.from_numpy(actions)
    value,log_prob,entropy = agent.policy.evaluate_actions(obs_tensor,actions_tensor_tmp)
    # print('value log prob entropy')
    # print(value,log_prob,entropy)
    obs_tmp = obs.copy()
    obs_tensor_tmp = obs_tensor.detach().clone()
    act_app = np.zeros(len(obs))
    image_name = 'occu_pushing' + ".png"
    #print(image_name)
    occu = occupancy.copy()
    occu[actions.flatten()[0],actions.flatten()[1]] = 255
    cv2.imwrite(path_save+image_name,occu)
    image_name = 'tsdf_pushing_1' + ".png"
    '''
    for j in range(3):
        obs_tmp = np.rot90(obs_tmp,1,(2,1))
        obs_tmp = obs_tmp.copy()
        obs_tensor_tmp = obs_tensor_tmp.rot90(1,[3,2])
        actions_tmp, _ = agent.predict(obs_tmp, deterministic=True)
        actions_tensor_tmp =  torch.from_numpy(actions_tmp)
        value_tmp,log_prob_tmp,entropy_tmp = agent.policy.evaluate_actions(obs_tensor_tmp,actions_tensor_tmp)
        for i in range(len(obs_tensor)):
            # if float(log_prob_tmp[i])>float(log_prob[i]):
            if float(value_tmp[i]) > float(value[i]):
                actions[i] = actions_tmp[i]
                act_app[i] = j * 2.0 +2.0
                log_prob[i] = log_prob_tmp[i]
                value[i] = value_tmp[i]
    actions_origin = actions.copy()
    for _ in range(len(obs)):
        if act_app[_] == 2:
            actions[_,0] = 49-actions[_,1]
            actions[_,1] = actions_origin[_,0]
        elif act_app[_] == 4:
            actions[_,0] = 49-actions[_,0]
            actions[_,1] = 49-actions_origin[_,1]
        elif act_app[_] == 6:
                actions[_,0] = actions[_,1]
                actions[_,1] = 49-actions_origin[_,0]
    '''
    '''
    for _ in range(len(value)):
            if float(value[_]) <=-0.1:
                act_app[_] = 10
    '''
    actions_new = np.c_[actions,act_app.T]   
    return actions_new 
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
class DepthTransformerTalker:
    def __init__(self):
        #rospy.init_node('depth_transformer_talker', anonymous=True)
        
        
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        '''
        self.image_pub_tsdf = rospy.Publisher('/tsdf_image_pushing', Image, queue_size=1)
        self.image_pub_occu = rospy.Publisher('/occu_image_pushing', Image, queue_size=1)
        '''
        self.occu_size = [50,50]
        self.save_path = './data/point_cloud/'
        # Subscribe to the depth image topic
        rospy.Subscriber('/fixed_camera/depth/camera_info', CameraInfo, self.camera_info_callback)
        rospy.Subscriber('/fixed_camera/depth/image_rect_raw', Image, self.depth_image_callback)
        
    
    def get_x_y(self):
        action=get_x_y(self.tsdf,self.occu)
        print('action')
        print(action)
        action = action.flatten()
        action_cat = [0.0,0.0]
        action_cat[1] = self.min_y + 0.01*action[0]
        action_cat[0] = self.min_x + 0.01*action[1]
        print(self.min_x,self.min_y)
        return action_cat
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        # return self.R
    def depth_pixel_to_camera(self,cv_image):
        #print(cv_image)
        [height, width] = cv_image.shape
        #print('image shape')
        #print(cv_image.shape)
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)        
        x = (u.flatten() - self.camera_cx)/self.camera_fx
        #print(x[:400])
        y = (v.flatten() - self.camera_cy)/self.camera_fy
        
        ##########################################
        ## TODO: switch between real and simulation
        #z = cv_image.flatten()/1000 ### real robot
        z = cv_image.flatten() ### simulation
        #print(z)
        ##########################################
        '''
        print('z')
        print(len(z),z[np.nonzero(z)].shape)
        print(len(z),np.nonzero(z))
        '''
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        #print(np.max(x),np.min(x),np.max(y),np.min(y),np.max(z),np.min(z))
        points=np.zeros((x.shape[0],4))
        points[:,3] = 1
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        points[:,2] = z.flatten()
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot ##################
        #### TODO:
        points = points[np.where(points[:,2]<1.2)]
        ####################################
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        
        points = points[np.where(points[:,0]<0.39)]
        points = points[np.where(points[:,0]>-0.11)]
        points = points[np.where(points[:,1]<0.11)]
        points = points[np.where(points[:,1]>-0.39)]
        
        ########################################
        
        
        
        ##########################################
        ## TODO: switch between real and simulation
        ############# for simulation #############
        '''
        points = points[np.where(points[:,0]<0.5)]
        points = points[np.where(points[:,0]>-0.11)]
        points = points[np.where(points[:,1]<0.11)]
        points = points[np.where(points[:,1]>-0.39)]
        '''
        ##########################################
        #print('num of points')
        #print(len(points))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        #print('show pcd')
        o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        return pcd, points
    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.camera_cx = self.camera_matrix[0,2]
        self.camera_cy = self.camera_matrix[1,2]
        self.camera_fx = self.camera_matrix[0,0]
        self.camera_fy = self.camera_matrix[1,1]
        self.distort_coeff = msg.D
        #print('self.camera_cx')
        #print(self.camera_cx)
    def depth_image_callback(self, msg):
        #print('get new image')
        self.msg = msg
        self.cv_image = self.bridge.imgmsg_to_cv2(self.msg, 'passthrough')
        print(np.min(self.cv_image))
        print('self.camera_cx')
        print(self.camera_cx)
        
    def get_images(self):
        
        # Transform pixel coordinates to camera coordinates
        # (Add your pixel to camera coordinate transformation code here)
        #cv_image = self.bridge.imgmsg_to_cv2(self.msg, 'passthrough')
        '''
        print('max cv image')
        print(np.max(cv_image),np.min(cv_image))
        print(cv_image)
        '''
        print('run get image')
        print(self.camera_cx)
        print(np.min(self.cv_image))
        pcd, points = self.depth_pixel_to_camera(self.cv_image)
        # Transform camera coordinates to world coordinates using tf
        camera_frame = "/fixed_camera_depth_optical_frame"  # Update with your camera frame
        world_frame = "/world"    # Update with your world frame

        (self.trans, self.quat) = self.listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
        #print('transformation and rotation')
        #print(self.quat,self.trans)
        # get the rotation matrix
        self.quaternion_to_matrix(self.quat)
        #print(self.R)
            
        world_points = points[:,:3].copy()
        pcd_w = self.camera_to_world(world_points)
        tsdf = self.pcd_to_tsdf()
        tsdf = tsdf.astype(np.uint8)
        '''
        tsdf = cv2.cvtColor(tsdf, cv2.COLOR_GRAY2BGR)
        tsdf_ros = self.bridge.cv2_to_imgmsg(tsdf, encoding="bgr8")
        '''
        # (Add your camera to world coordinate transformation code here)
        occu = self.pcd_to_occu()
        occu = occu.astype(np.uint8)
        image_name = 'occu_pushing' + ".png"
        #print(image_name)
        cv2.imwrite(path_save+image_name,occu)
        image_name = 'tsdf_pushing' + ".png"
        print(image_name)
        print(path_save+image_name)
        cv2.imwrite(path_save+image_name,tsdf)
        occu_ros = self.bridge.cv2_to_imgmsg(occu)
        self.occu = occu.copy()
        self.tsdf = tsdf.copy()
        return occu, tsdf
        # Now you can use the transformed depth image data as needed

        # Print the transformed depth image data (replace this with your logic)
        #rospy.loginfo("Transformed Depth Image Data: %s", transformed_depth_image_data)

        # Publish the transformed depth image
        '''
        self.image_pub_tsdf.publish(tsdf_ros)
        self.image_pub_occu.publish(occu_ros)
        '''
    def camera_to_world(self,world_points):
        for i in range(len(world_points)):
            world_points[i] = np.dot(self.R,world_points[i])+self.trans
        print('world limit')
        print(np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
        self.max_x = np.max(world_points[:,0])
        self.min_x = np.min(world_points[:,0])
        self.max_y = np.max(world_points[:,1])
        self.min_y = np.min(world_points[:,1])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        #o3d.io.write_point_cloud(self.save_path+"pcd_world_frame.ply", pcd)
        obb = pcd.get_oriented_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()
        obb.color = [1,0,0]
        aabb.color = [0,0,0]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        #o3d.visualization.draw_geometries([pcd,obb,aabb,axes])
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        self.inlier_cloud = pcd.select_by_index(inliers)
        # Extract outliers (non-plane points)
        self.outlier_cloud = pcd.select_by_index(inliers, invert=True)
        np_outlier_cloud = np.array(self.outlier_cloud.points)
        np_outlier_cloud[:,2] = 0
        self.outlier_cloud = o3d.geometry.PointCloud()
        self.outlier_cloud.points = o3d.utility.Vector3dVector(np_outlier_cloud)
        # Visualize the results
        #o3d.visualization.draw_geometries([self.inlier_cloud])
        #o3d.visualization.draw_geometries([self.outlier_cloud])
        
        return pcd
    def pcd_to_tsdf(self):
    
        Nx, Ny = self.occu_size
        x = np.linspace(self.min_x, self.max_x, Nx)
        y = np.linspace(self.min_y, self.max_y, Ny)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        distance = pts_grid.compute_point_cloud_distance(self.outlier_cloud)
        dist = np.array(distance)
        Tsdf = dist.reshape(Ny,Nx)
        Tsdf = Tsdf*255
        Tsdf = np.fliplr(Tsdf)
        Tsdf = np.rot90(Tsdf)
        Tsdf = Tsdf.copy()
        #print(np.max(Tsdf),np.min(Tsdf))
        return Tsdf
    def pcd_to_occu(self):
        Nx, Ny = self.occu_size
        pts = np.asarray(self.outlier_cloud.points)
        u = (pts[:,0] - self.min_x)/ ( self.max_x-self.min_x )
        v = (pts[:,1] - self.min_y)/ ( self.max_y-self.min_y )
        u = (Nx-1)*u
        v = (Ny-1)*v
        occupancy = np.zeros( (Ny,Nx) )
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        u_ind = np.where(u<Nx)
        u = u[u_ind]
        v = v[u_ind]
        v_ind = np.where(v<Ny)
        u = u[v_ind]
        v = v[v_ind]
        u_ind = np.where(u>=0)
        u = u[u_ind]
        v = v[u_ind]
        v_ind = np.where(v>=0)
        u = u[v_ind]
        v = v[v_ind]
        occupancy[v,u] = 1
        occupancy = occupancy*255
        occupancy = np.fliplr(occupancy)
        occupancy = np.rot90(occupancy)
        occupancy = occupancy.copy()
        return occupancy
    def decode_image_pos(self):
        pass
    
'''
class ImpedancePushing:
    def __init__(self):
        ImpedancePushingPublisher = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
        rospy.Subscriber('/cartesian_impedance_example_controller/equilibrium_pose',PoseStamped, self.ImpedancePushingcallback)
    def ImpedancePushingcallback(self, msg):
        print('ImpedanceControl')
        print(msg)
'''       
class PerformPushing(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(PerformPushing, self).__init__()

        ## BEGIN_SUB_TUTORIAL setup
        ##
        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("perform_pushing", anonymous=True)

        ## Instantiate a `RobotCommander`_ object. Provides information such as the robot's
        ## kinematic model and the robot's current joint states
        robot = moveit_commander.RobotCommander()

        ## Instantiate a `PlanningSceneInterface`_ object.  This provides a remote interface
        ## for getting, setting, and updating the robot's internal understanding of the
        ## surrounding world:
        scene = moveit_commander.PlanningSceneInterface()
	
        ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
        ## to a planning group (group of joints).  In this tutorial the group is the primary
        ## arm joints in the Panda robot, so we set the group's name to "panda_arm".
        ## If you are using a different robot, change this value to the name of your robot
        ## arm planning group.
        ## This interface can be used to plan and execute motions:
        group_name = "panda_manipulator"
        move_group = moveit_commander.MoveGroupCommander(group_name)

        ## Create a `DisplayTrajectory`_ ROS publisher which is used to display
        ## trajectories in Rviz:
        display_trajectory_publisher = rospy.Publisher(
            "/move_group/display_planned_path_pushing",
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20,
        )

        ## END_SUB_TUTORIAL

        ## BEGIN_SUB_TUTORIAL basic_info
        ##
        ## Getting Basic Information
        ## ^^^^^^^^^^^^^^^^^^^^^^^^^
        # We can get the name of the reference frame for this robot:
        planning_frame = move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)

        # We can also print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)

        # We can get a list of all the groups in the robot:
        group_names = robot.get_group_names()
        print("============ Available Planning Groups:", robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("============ Printing robot state")
        print(robot.get_current_state())
        print("")
        ## END_SUB_TUTORIAL

        # Misc variables
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
        ###################################################################
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
    def get_gripper_pose(self):
        current_pose = self.move_group.get_current_pose().pose
        print('current pose')
        print(current_pose)
    def go_to_joint_state(self):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group

        ## BEGIN_SUB_TUTORIAL plan_to_joint_state
        ##
        ## Planning to a Joint Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^^
        ## The Panda's zero configuration is at a `singularity <https://www.quora.com/Robotics-What-is-meant-by-kinematic-singularity>`_, so the first
        ## thing we want to do is move it to a slightly better configuration.
        ## We use the constant `tau = 2*pi <https://en.wikipedia.org/wiki/Turn_(angle)#Tau_proposals>`_ for convenience:
        # We get the joint values from the group and change some of the values:
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 3
        joint_goal[4] = 0
        joint_goal[5] = tau / 4  # 1/6 of a turn
        joint_goal[6] = tau / 8
        
        #joint_goal[0] = 0
        #joint_goal[1] = -tau / 8
        #joint_goal[2] = 0
        #joint_goal[3] = -tau / 4
        #joint_goal[4] = 0
        #joint_goal[5] = tau / 6  # 1/6 of a turn
        #joint_goal[6] = 0

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        move_group.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        move_group.stop()

        ## END_SUB_TUTORIAL

        # For testing:
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self,x=0.5,y = 0.0,rotate=0):
        # Copy class variables to local variables to make the web tutorials more clear.
        # In practice, you should use the class variables directly unless you have a good
        # reason not to.
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
        ## BEGIN_SUB_TUTORIAL plan_to_pose
        ##
        ## Planning to a Pose Goal
        ## ^^^^^^^^^^^^^^^^^^^^^^^
        ####################################### move to point above target pose ###########################
        ## We can plan a motion for this group to a desired pose for the
        ## end-effector:
        pose_goal = geometry_msgs.msg.Pose()
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        #pose_goal.orientation.z = 0.0
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = 0.32
        
        gripper_frame = "/panda_hand_tcp"   # Update with your camera frame
        ee_frame = self.eef_link    # Update with your world frame

        (self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)

        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.01):
            print('cannot find path to move the gripper above the targte point')
            return all_close(pose_goal, current_pose, 0.01)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if rotate == 1:
            if joint_goal[6]< 2:
                joint_goal[6]+=tau / 4
            else:
                joint_goal[6]-=tau / 4
            move_group.go(joint_goal, wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            move_group.stop()

            ## END_SUB_TUTORIAL

            # For testing:
            current_joints = move_group.get_current_joint_values()
            if not all_close(joint_goal, current_joints, 0.01):
                print('ubable to rotate joint 6')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        
        pose_goal.position.z = 0.22
        move_group.set_pose_target(pose_goal)
        ## Now, we call the planner to compute the plan and execute it.
        # `go()` returns a boolean indicating whether the planning and execution was successful.
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        ## END_SUB_TUTORIAL

        # For testing:
        # Note that since this section of code will not be included in the tutorials
        # we use the class variable rather than the copied state variable
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.01):
            print('cannot find path to move the gripper to the targte point')
        return all_close(pose_goal, current_pose, 0.01)

def main():
    try:
        pushing = PerformPushing()
        input( "============ Press `Enter` to execute a movement using a joint state goal ...")       
        pushing.go_to_joint_state()
        input( "============ Press `Enter` to continue ...")    
        depth_image_transformer_talker = DepthTransformerTalker()
        input( "============ Press `Enter` to continue ...")   
        gripper = PandaGripperClient()
    except rospy.ROSInterruptException:
        pass

    rate = rospy.Rate(1)
    
    while not rospy.is_shutdown():
        try:
            
            depth_image_transformer_talker = DepthTransformerTalker()
            #input("============ Press `Enter` to move to target point ...")
            depth_image_transformer_talker.get_images()
            action=depth_image_transformer_talker.get_x_y()
            print('world action')
            print(action)
            pushing.go_to_pose_goal(x=action[0],y = action[1],rotate=1)
            rospy.sleep(2)
            gripper.grasp()
            rospy.sleep(2)
            pushing.get_gripper_pose()
            pushing.go_to_joint_state()
            
        except rospy.ROSInterruptException:
            return
        except KeyboardInterrupt:
            return
        rate.sleep()
    roscpp_shutdown()
if __name__ == "__main__":
    main()
