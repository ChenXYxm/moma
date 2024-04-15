#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import open3d as o3d
path = './data/occu/'
def get_tsdf(occu):
    shape_occu = occu.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    points_np = np.zeros((Nx*Ny,3)).astype(np.float32)
    for i in range(Nx):
        for j in range(Ny):
            num_id = Nx*i + j
            points_np[num_id][1] = 0.5+0.01*i
            points_np[num_id][0] = 0.5+0.01*j
            if occu[i,j] == 1:
                points_np[num_id][2] = 0.015
    points_obj_id = np.where(points_np[:,2]>=0.01)
    #print('points_obj_id')
    #print(points_obj_id)
    points_obj = points_np[points_obj_id[0]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_obj)
    x = np.linspace(0.5, 1.0, Nx)
    y = np.linspace(0.5, 1.0, Ny)
    xv, yv = np.meshgrid(x, y)

    grid = np.zeros((Nx*Ny,3))
    grid[:,0] = xv.flatten()
    grid[:,1] = yv.flatten()
    pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
    distance = pts_grid.compute_point_cloud_distance(pcd)
    dist = np.array(distance)
    # norm = mplc.Normalize(vmin=min(distance), vmax=max(distance), clip=True)
    Tsdf = dist.reshape(Ny,Nx)
    # Tsdf = np.fliplr(Tsdf)
    # plt.imshow(Tsdf)
    # plt.show()
    # print('pointcloud 3d')
    # print(pointcloud_w)
    # o3d.visualization.draw_geometries([pcd])
    Tsdf = Tsdf*255
    return Tsdf
class DepthImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()

        # Create a subscriber for the depth image
        self.depth_sub = rospy.Subscriber("/fixed_camera/depth/image_rect_raw", Image, self.depth_image_callback)

        # Create a publisher for the processed image
        self.processed_image_pub = rospy.Publisher("/fixed_camera/depth/processed_image", Image, queue_size=10)
        self.processed_tsdf_pub = rospy.Publisher("/fixed_camera/depth/processed_tsdf", Image, queue_size=10)
    def depth_image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Process the depth image (add your processing steps here)
            processed_image = self.process_depth_image(depth_image)

            # Convert the processed image back to ROS Image message
            processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image)
            tsdf = self.process_tsdf(processed_image)
            tsdf_msg = self.bridge.cv2_to_imgmsg(tsdf)
            
            # Publish the processed image
            self.processed_image_pub.publish(processed_image_msg)
            self.processed_tsdf_pub(tsdf_msg)

        except Exception as e:
            print(e)

    def process_depth_image(self, depth_image):
        # Add your image processing steps here
        # For example, you can perform operations like cropping, flipping, etc.
        # Ensure the processed image has the same size and encoding as the original depth image.
        processed_image = depth_image  # Placeholder for now
        #print(np.max(depth_image),np.min(depth_image),np.mean(depth_image))
        # Display the depth image
        #cv2.imshow("Depth Image", depth_image)
        #cv2.waitKey(1)
        print(depth_image.shape)
        image_tmp = np.array(depth_image)
        #print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
        #image_tmp=np.where(image_tmp < 1000.0, image_tmp, 1000.0)
        image_tmp=np.where(image_tmp >= 1.1, image_tmp, 0)
        image_tmp=np.where(image_tmp < 1.1, image_tmp, 255)
        
        #image_tmp = image_tmp.astype(np.uint8)
        image_tmp = np.array(image_tmp[80:220,320:460])
        new_size = (50,50)
        image_tmp = cv2.resize(image_tmp,new_size)
        image_tmp=np.where(image_tmp <100, 1, image_tmp)
        image_tmp=np.where(image_tmp >=100, 0, image_tmp)
        print(image_tmp.shape)
        #print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
        image_tmp = image_tmp*255
        return image_tmp
    def process_tsdf(self,processed_image):
        image_tmp = np.array(processed_image)
        if np.max(image_tmp)>0:
            image_tmp = image_tmp/np.max(image_tmp)
        Tsdf = get_tsdf(image_tmp)
        return Tsdf
def main():
    rospy.init_node('depth_image_processor', anonymous=True)
    depth_processor = DepthImageProcessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
    

