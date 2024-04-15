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
        cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
        print(cv_image.shape)
        #cv2.imshow("Image", cv_image)
        #cv2.waitKey(0)
        timestr = "%.6f" % msg.header.stamp.to_sec()
        i +=1 
        print(i)
        image_name = str(i) + ".png"
        cv2.imwrite(path+image_name,cv_image)
        print(np.max(cv_image),np.min(cv_image))
        
        [height, width] = cv_image.shape
        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        x = (u.flatten() - camera_cx)/camera_fx
        #print(x[:400])
        y = (v.flatten() - camera_cy)/camera_fy
        #z = cv_image.flatten() / 1000
        z = cv_image.flatten()
        x = np.multiply(x,z)
        y = np.multiply(y,z)
        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
        print(np.max(x),np.min(x),np.max(y),np.min(y))
        points=np.zeros((x.shape[0],4))
        points[:,3] = 1
        points[:,0] = x.flatten()
        points[:,1] = y.flatten()
        points[:,2] = z.flatten()
        points = points[np.where(points[:,0]<0.41)]
        points = points[np.where(points[:,0]>-0.11)]
        points = points[np.where(points[:,1]<0.11)]
        points = points[np.where(points[:,1]>-0.39)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        world_points = points[:,:3].copy()
        for i in range(len(world_points)):
            world_points[i] = np.dot(rot_m,world_points[i])+transform_arr
        print(np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world_points)
        obb = pcd.get_oriented_bounding_box()
        aabb = pcd.get_axis_aligned_bounding_box()
        obb.color = [1,0,0]
        aabb.color = [0,0,0]
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd,obb,aabb,axes])
        proj_points = world_points - aabb.get_center()
        height_map = np.zeros((200, 200))
        proj_points += np.abs(proj_points.min(axis=0))
        proj_points *= 200 / proj_points.max(axis=0)
        proj_points = proj_points.astype(int)
        proj_points[proj_points < 0] = 0
        proj_points[proj_points >= 200] = 199
        print(np.max(points[:, 2]),np.min(points[:, 2]))
        points[:, 2] = (points[:, 2]-np.min(points[:, 2]))*255/(np.max(points[:, 2])-np.min(points[:, 2]))
        height_map[proj_points[:, 0], proj_points[:, 1]] = points[:, 2]
        height_map.astype(np.uint8)
        image_name = 'heightmap'+str(i) + ".png"
        print(image_name)
        cv2.imwrite(path+image_name,height_map)
        if i > 11:
            break
