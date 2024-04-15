#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_geometry_msgs import PointStamped
from geometry_msgs.msg import Point

def depth_to_world_coordinates(depth_msg, camera_info_msg):
    # Convert depth_msg to a numpy array
    depth_image = CvBridge().imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

    # Assuming K is the camera intrinsic matrix (fx, fy, cx, cy)
    fx = camera_info_msg.K[0]
    fy = camera_info_msg.K[4]
    cx = camera_info_msg.K[2]
    cy = camera_info_msg.K[5]

    # Create a point cloud to store world coordinates
    world_point_cloud = []

    # Iterate through each pixel in the depth image
    for u in range(depth_msg.width):
        for v in range(depth_msg.height):
            # Convert pixel coordinates to camera coordinates
            x_camera = (u - cx) * depth_image[v, u] / fx
            y_camera = (v - cy) * depth_image[v, u] / fy
            z_camera = depth_image[v, u]

            # Use tf2 to get the transformation from camera frame to world frame
            transform_listener = tf2_ros.TransformListener(tf2_ros.Buffer())
            transform_listener.waitForTransform("world", "fixed_camera_depth_frame", rospy.Time(), rospy.Duration(4.0))
            trans, rot = transform_listener.lookupTransform("world", "fixed_camera_depth_frame", rospy.Time())

            # Apply the transformation to get world coordinates
            world_point = np.dot(np.array([[x_camera, y_camera, z_camera, 1.0]]), tf2_ros.Transform(trans, rot).matrix)
            world_point_cloud.append(world_point.flatten()[:3])  # Extract x, y, z

    return world_point_cloud

def main():
    rospy.init_node("depth_to_world_coordinates_node")

    # Create a publisher for world coordinates
    world_coordinates_pub = rospy.Publisher("/world_coordinates", PointStamped, queue_size=10)

    # Subscribe to the depth image and camera info topics
    rospy.Subscriber("/fixed_camera/depth/image_raw", Image, depth_callback)
    rospy.Subscriber("/fixed_camera/color/camera_info", CameraInfo, camera_info_callback)

    rospy.spin()

if __name__ == "__main__":
    main()

