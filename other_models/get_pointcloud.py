import rosbag
import cv2
from sensor_msgs import point_cloud2 
import numpy as np
import open3d as o3d
import pickle as pkl
bag_file = './data/points_blocks.bag'

bag = rosbag.Bag(bag_file,'r')
bag_data = bag.read_messages()
path = './data/point_cloud/'
i = 0
for topic, msg, t in bag_data:
    print(t)
    print(topic)
    if topic == '/fixed_camera/depth/color/points':
        points = point_cloud2.read_points(msg,field_names=("x","y","z"))
        #p = point_cloud2.read_points_list(msg)
        i += 5
        xyz = np.array(list(points))
        p_o3d = o3d.geometry.PointCloud()
        p_o3d.points = o3d.utility.Vector3dVector(xyz)
        filename = path + "output_pointcloud"+str(i)+".pkl"
        fileObject = open(filename, 'wb')
        pkl.dump(xyz, fileObject)
        fileObject.close()
        #p_np = np.asarray(p_o3d)
        #print(np.max(p_np[:,2]))
        #print(p_o3d)
        o3d.visualization.draw_geometries([p_o3d])
        print(i)
        if i >10:
            break
        #o3d.io.write_point_cloud(path+"output_pointcloud"+str(i)+".ply", p_o3d)
        #print(p)
