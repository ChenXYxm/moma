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
from stable_baselines3 import PPO
import torch
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import tf2_ros
from moma_utils.ros.panda import PandaGripperClient
import sys
#import matplotlib.pyplot as plt
#import tkinter as tk

try:
    from math import pi, tau, dist, fabs, cos
except:  # For Python 2 compatibility
    from math import pi, fabs, cos, sqrt

    tau = 2.0 * pi

    def dist(p, q):
        return sqrt(sum((p_i - q_i) ** 2.0 for p_i, q_i in zip(p, q)))

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


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
    
    #print(image_name)
    
    
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
    occu = occupancy.copy()
    occu[actions.flatten()[0],actions.flatten()[1]] = 255
    image_name = "./data/point_cloud/tsdf_pushing_new.png"
    cv2.imwrite(image_name,occu)
    '''
    for _ in range(len(value)):
            if float(value[_]) <=-0.1:
                act_app[_] = 10
    '''
    actions_new = np.c_[actions,act_app.T]   
    return actions_new 


class PerformPushing(object):
    """MoveGroupPythonInterfaceTutorial"""

    def __init__(self):
        super(PerformPushing, self).__init__()


        ## First initialize `moveit_commander`_ and a `rospy`_ node:
        moveit_commander.roscpp_initialize(sys.argv)
        #rospy.init_node("perform_pushing", anonymous=True)

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

        
        planning_frame = move_group.get_planning_frame()
        #print("============ Planning frame: %s" % planning_frame)

        # print the name of the end-effector link for this group:
        eef_link = move_group.get_end_effector_link()
        #print("============ End effector link: %s" % eef_link)

        # list of all the groups in the robot:
        group_names = robot.get_group_names()
        #print("============ Available Planning Groups:", robot.get_group_names())
          
        #print("============ Printing robot state")
        #print(robot.get_current_state())
        #print("")
        
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

        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 8
        joint_goal[2] = 0
        joint_goal[3] = -tau / 3
        joint_goal[4] = 0
        joint_goal[5] = tau / 4  # 1/6 of a turn
        joint_goal[6] = tau / 8
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)

    def go_to_pose_goal(self,x=0.5,y = 0.0,rotate=0):
       
        move_group = self.move_group
        #move_group.set_end_effector_link(self.true_eef)
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
        pose_goal.position.z = 0.4
        
        #gripper_frame = "/panda_hand_tcp"   # Update with camera frame
        #ee_frame = self.eef_link    # Update with world frame

        #(self.trans, self.quat) = self.listener.lookupTransform(ee_frame, gripper_frame, rospy.Time(0))

        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper above the target point')
            return all_close(pose_goal, current_pose, 0.05)
        #########################################################################
        ############################ rotate the gripper #########################
        joint_goal = move_group.get_current_joint_values()
        if rotate == 0 or rotate==2:
            if joint_goal[6]< 2:
                joint_goal[6]+=tau / 4
            else:
                joint_goal[6]-=tau / 4
            move_group.go(joint_goal, wait=True)
            move_group.stop()
            current_joints = move_group.get_current_joint_values()
            if not all_close(joint_goal, current_joints, 0.05):
                print('ubable to rotate joint 6')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.25
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.17
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        ################################ push ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y -=0.035
        elif rotate==1:
            pose_goal.position.x +=0.035
        elif rotate==2:
            pose_goal.position.y +=0.035
        elif rotate==3:
            pose_goal.position.x -=0.035
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)

        move_group.stop()

        move_group.clear_pose_targets()
        ################################ push ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y -=0.035
        elif rotate==1:
            pose_goal.position.x +=0.035
        elif rotate==2:
            pose_goal.position.y +=0.035
        elif rotate==3:
            pose_goal.position.x -=0.035
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        print('success: ', success)
        move_group.stop()

        move_group.clear_pose_targets()
        ##########################################################################
        
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: pushing')
        ################################ return ####################################
        pose_goal = self.move_group.get_current_pose().pose
        if rotate == 0:
            pose_goal.position.y +=0.03
        elif rotate==1:
            pose_goal.position.x -=0.03
        elif rotate==2:
            pose_goal.position.y -=0.03
        elif rotate==3:
            pose_goal.position.x +=0.03
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        ############################################
        pose_goal = self.move_group.get_current_pose().pose
        pose_goal.position.z +=0.1
        
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)

        move_group.stop()

        move_group.clear_pose_targets()
        ##########################################################################
        return all_close(pose_goal, current_pose, 0.05)

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


class DepthImageTransformerTalker:
    def __init__(self):
        rospy.init_node('pushing_obj', anonymous=True)
        self.bridge = CvBridge()
        self.listener = tf.TransformListener()
        self.image_pub = rospy.Publisher('/transformed_depth_image', Image, queue_size=1)
        self.image_pub_occu = rospy.Publisher('/occu_image', Image, queue_size=1)
        self.occu_size = [50,50]
        self.save_path = './data/point_cloud/'
        self.puhsing_controller = PerformPushing()
        self.gripper = PandaGripperClient()
        #self.occu = np.zeros((50,50))
        # Subscribe to the depth image topic
        self.puhsing_controller.go_to_joint_state()
        self.gripper.grasp()
        rospy.Subscriber('/fixed_camera/depth/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber('/fixed_camera/depth/image_rect_raw', Image, self.depth_image_callback, queue_size=1,buff_size=2**30) ##307200
        self.debug_pub = rospy.Publisher('/debug_depth', Image, queue_size=1)
        self.time = 0.0
        #rospy.spin()
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        # return self.R
    def depth_pixel_to_camera(self,cv_image):
        [height, width] = cv_image.shape
        
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)        
        x = (u.flatten() - self.camera_cx)/self.camera_fx
        #print(x[:400])
        y = (v.flatten() - self.camera_cy)/self.camera_fy
        
        ##########################################
        ## TODO: switch between real and simulation
        z = cv_image.flatten()/1000 ### real robot
        #z = cv_image.flatten() ### simulation
        ##########################################
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
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        #points = points[np.where(points[:,0]<0.395)]
        #points = points[np.where(points[:,0]>-0.22)]
        #points = points[np.where(points[:,1]<0.29)]
        #points = points[np.where(points[:,1]>-0.325)]
        ########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        #o3d.visualization.draw_geometries([pcd])
        
        
        ##########################################
        ## TODO: switch between real and simulation
        ############# for simulation #############
        points = points[np.where(points[:,0]<0.327)]
        points = points[np.where(points[:,0]>-0.173)]
        points = points[np.where(points[:,1]<0.223)]
        points = points[np.where(points[:,1]>-0.277)]
        ##########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        #o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        return pcd, points
    def image_processing(self,image):
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(image,-1,kernel)
        img_blur=np.where(img_blur <100, 0, img_blur)
        img_blur=np.where(img_blur >=100, 255, img_blur)
        cv2.imwrite(self.save_path+'occu_filtered_new.png',img_blur)
        return img_blur
    def camera_info_callback(self,msg):
        self.camera_matrix = np.array(msg.K).reshape((3, 3))
        self.camera_cx = self.camera_matrix[0,2]
        self.camera_cy = self.camera_matrix[1,2]
        self.camera_fx = self.camera_matrix[0,0]
        self.camera_fy = self.camera_matrix[1,1]
        #print('self.camera_cx')
        #print(self.camera_cx)
    def camera_to_world(self,world_points):
        for i in range(len(world_points)):
            world_points[i] = np.dot(self.R,world_points[i])+self.trans
        #print(np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
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
        #o3d.visualization.draw_geometries([pcd,aabb,axes])
        o3d.io.write_point_cloud(self.save_path+"pcd_world2.ply", pcd)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        self.inlier_cloud = pcd.select_by_index(inliers)
        # Extract outliers (non-plane points)
        self.outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # Visualize the results
        #o3d.visualization.draw_geometries([self.inlier_cloud])
        #o3d.visualization.draw_geometries([self.outlier_cloud])
        outlier_cloud_np = np.array(self.outlier_cloud.points)
        outlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model
        for i in range(len(outlier_cloud_np)):
            x,y,z = outlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                outlier_cloud_list.append(outlier_cloud_np[i])
        outlier_cloud = o3d.geometry.PointCloud()    
        outlier_cloud_np = np.array(outlier_cloud_list).reshape(-1,3) 
        outlier_cloud_np[:,2] = 0 
        outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_np)
        #o3d.visualization.draw_geometries([outlier_cloud])
        self.outlier_cloud = outlier_cloud
        # Visualize the results
        
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
        return occupancy
    def decode_image_pos(self):
        pass
    def depth_image_callback(self, msg):
        msg_time = msg.header.stamp.to_sec()
        current_time = rospy.Time.now().to_sec()
        print("Delay is ", current_time - msg_time)
        if current_time - msg_time < 2.0:
            try:
                self.debug_pub.publish(msg)
                # Transform pixel coordinates to camera coordinates
                print(msg.header.stamp)
                print(msg.header.seq)
                print(msg.header.frame_id)
                msg_time = msg.header.stamp.to_sec()
                current_time = rospy.Time.now().to_sec()
                print("Delay is ", current_time - msg_time)
            
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                print(cv_image.shape)
            
                #plt.imshow(cv_image)
                cv_image_tmp = cv_image.copy()
                cv_image_tmp[np.where(cv_image_tmp<=900)] = 0
                cv_image_tmp[np.where(cv_image_tmp>900)] = 255
                cv_image_tmp = cv_image_tmp.astype(np.uint8)
                cv2.imwrite(self.save_path+'depth.png',cv_image_tmp)
                pcd, points = self.depth_pixel_to_camera(cv_image)
                # Transform camera coordinates to world coordinates using tf
                camera_frame = "/fixed_camera_depth_optical_frame"  # Update with your camera frame
                world_frame = "/world"    # Update with your world frame

                (self.trans, self.quat) = self.listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
                #print(self.quat,self.trans)
                # get the rotation matrix
                self.quaternion_to_matrix(self.quat)
            
                world_points = points[:,:3].copy()
                pcd_w = self.camera_to_world(world_points)
                tsdf = self.pcd_to_tsdf()
                tsdf_tmp = tsdf.copy()
                tsdf = tsdf.astype(np.uint8)
                tsdf = cv2.cvtColor(tsdf, cv2.COLOR_GRAY2BGR)
                #tsdf_ros = self.bridge.cv2_to_imgmsg(tsdf, encoding="bgr8")
            
                occu = self.pcd_to_occu()
                occu = self.image_processing(occu)
                occu_tmp = occu.copy()
                occu = occu.astype(np.uint8)
                #occu_ros = self.bridge.cv2_to_imgmsg(occu)

                self.occu = occu.copy()
           
                # Print the transformed depth image data (replace this with your logic)
                #rospy.loginfo("Transformed Depth Image Data: %s", transformed_depth_image_data)

                # Publish the transformed depth image
                #self.image_pub.publish(tsdf_ros)
                #self.image_pub_occu.publish(occu_ros)
                cv2.imwrite(self.save_path+'occu_new.png',occu)
                cv2.imwrite(self.save_path+'tsdf_new.png',tsdf)
                action=get_x_y(tsdf_tmp,occu_tmp)
            
                #print('action: ',action)
                x,y,rot=self.transfrom_action_world(action)
            
                #input( "============ Press `Enter` to continue ...") 
                self.puhsing_controller.go_to_pose_goal(x=x,y = y,rotate=rot)
                #rospy.sleep(3)
            
                self.puhsing_controller.go_to_joint_state()
                #rospy.sleep(3)
                '''
                occu = self.get_occu()
                print('max occu')
                print(np.max(occu))
            
                '''
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                print("error")
                pass
    def transfrom_action_world(self,action):
        Nx, Ny = self.occu_size
        x,y,rot = action.flatten()
        rot = rot/2
        print(x,y,rot)
        x = self.min_x+x*(self.max_x-self.min_x)/float(Nx) 
        y = self.min_y+y*(self.max_y-self.min_y)/float(Nx) 
        #print(self.max_x,self.min_x,self.max_y,self.min_y,Nx)
        
        print("action world position: ",x,y,rot)
        return x,y,rot
    def get_occu(self):
        return self.occu
if __name__ == '__main__':
    try:
        depth_image_transformer_talker = DepthImageTransformerTalker()
        rospy.spin()
        occu = depth_image_transformer_talker.get_occu()
        print('max occu')
        print(np.max(occu))
        
    except rospy.ROSInterruptException:
        pass

