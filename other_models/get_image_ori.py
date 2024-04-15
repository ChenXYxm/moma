import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d

path = './data/depth_pic/'
bag_file = './data/depth_simulation.bag'
print(path)
bag = rosbag.Bag(bag_file,'r')
bag_data = bag.read_messages()

bridge = CvBridge()
i = 10
def quaternion_to_matrix(q):
    x, y, z,w  = q
    R = np.array([
    [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
    [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
    ])
    return R
'''following parameters are only for simulation'''
###################################################
camera_K = np.array([[602.1849879340944, 0.0, 320.5],
                      [0.0, 602.1849879340944, 240.5],
                       [0.0, 0.0, 1.0]])

camera_cx = camera_K[0,2]
camera_cy = camera_K[1,2]
camera_fx = camera_K[0,0]
camera_fy = camera_K[1,1]
quat = np.array([0.708, 0.700, -0.053, -0.075])

transform_arr = np.array([0.876, -0.179, 1.284])
####################################################
quat = quat/np.linalg.norm(quat)
rot_m = quaternion_to_matrix(quat)

for topic, msg, t in bag_data:
    print(t)
    print(topic)
    print(msg.header.frame_id)
    if topic == '/fixed_camera/depth/image_rect_raw':
        cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough') #16UC1
        print(cv_image.shape)
        #cv2.imshow("Image", cv_image)
        #cv2.waitKey(0)
        timestr = "%.6f" % msg.header.stamp.to_sec()
        i +=1 
        print(i)
        image_name = str(i) + ".png"
        cv2.imwrite(path+image_name,cv_image)
        print(np.max(cv_image),np.min(cv_image))
        
        
        if i > 11:
            break
