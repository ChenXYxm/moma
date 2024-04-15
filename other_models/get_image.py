import rosbag
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import open3d as o3d
import pickle
path = './data/'
#bag_file = './data/compare_with.bag'
bag_file = './data/pico_compare1.bag'
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
    print(topic)
    #if topic == '/camera/depth/camera_info':
    if topic == '/pico_flexx/camera_info':
        camera_info = msg
        print(camera_info.K,camera_info.D)
        break
camera_K = np.array(camera_info.K).reshape(3,3)
print(camera_K)
camera_cx = camera_K[0,2]
camera_cy = camera_K[1,2]
camera_fx = camera_K[0,0]
camera_fy = camera_K[1,1]
def distance_to_plane(point, plane_coefficients):
    A, B, C, D = plane_coefficients
    X, Y, Z = point
    numerator = np.abs(A * X + B * Y + C * Z + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    distance = numerator / denominator
    return distance
for topic, msg, t in bag_data:
    print(t)
    print(topic)
    print(msg.header.frame_id)
    ####################################
    #if topic == '/camera/depth/image_rect_raw':
    if topic == '/pico_flexx/image_depth':
    #########################
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
        #print(u.shape,u.flatten().shape)
        points_distorted = np.concatenate((u.flatten().reshape(-1,1), v.flatten().reshape(-1,1)), axis=1)
        #print(points_distorted)
        points_distorted = points_distorted.reshape((-1,1,2))
        #print(points_distorted)
        undistorted_points = cv2.undistortPoints(points_distorted, camera_K, np.array(camera_info.D).reshape(1,-1))
        
        print(np.array(camera_info.D).reshape(1,-1))
        print(undistorted_points)
        print(np.max(undistorted_points[:,0,0]),np.min(undistorted_points[:,0,0]),np.max(undistorted_points[:,0,1]),np.min(undistorted_points[:,0,1]))
        #for u_i in range(len(u)):
            #points = np.array([[u[u_i], v[u_i]]], dtype=np.float32)
            #undistorted_points = cv2.undistortPoints(points, camera_info.K, camera_info.D)
            #print(undistorted_points)
            #u_undistorted = undistorted_points[0, 0, 0]
            #v_undistorted = undistorted_points[0, 0, 1]
    
        x = undistorted_points[:,0,0]
        #print(x[:400])
        y = undistorted_points[:,0,1]
        
        print(np.max(x),np.min(x),np.max(y),np.min(y))
        ###############################################
        #z = cv_image.flatten() / 1000 ## realsense
        z = cv_image.flatten() ## pico and simulation
        ###############################################
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        ############################################
        ''' real sense'''
        '''
        points = points[np.where(points[:,0]<0.3)]
        points = points[np.where(points[:,0]>-0.2)]
        points = points[np.where(points[:,1]<0.16)]
        points = points[np.where(points[:,1]>-0.34)]
        '''
        '''pico'''
        
        points = points[np.where(points[:,0]<0.08)]
        points = points[np.where(points[:,0]>-0.42)]
        points = points[np.where(points[:,1]<0.16)]
        points = points[np.where(points[:,1]>-0.34)]
        
        ############################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        inlier_cloud = pcd.select_by_index(inliers)
        low_bound = np.mean(np.array(inlier_cloud.points)[:,2])
        print('plane_model')
        print(plane_model)
        print(np.array(inlier_cloud.points))
        
        # Extract outliers (non-plane points)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        high_bound = np.mean(np.array(outlier_cloud.points)[:,2])-0.1
        o3d.visualization.draw_geometries([outlier_cloud])
        outlier_cloud_np = np.array(outlier_cloud.points)
        outlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model
        for i in range(len(outlier_cloud_np)):
            x,y,z = outlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator < 0 and distance<0.3:
                outlier_cloud_list.append(outlier_cloud_np[i])
            
        outlier_cloud_np = np.array(outlier_cloud_list).reshape(-1,3)   
        #outlier_cloud_np = outlier_cloud_np[np.where(outlier_cloud_np[:,2]<low_bound)]
        #outlier_cloud_np = outlier_cloud_np[np.where(outlier_cloud_np[:,2]>high_bound)]
        print('lower bound')
        print(low_bound, np.mean(outlier_cloud_np[:,2]))
        outlier_cloud = o3d.geometry.PointCloud()
        outlier_cloud.points = o3d.utility.Vector3dVector(outlier_cloud_np)
        o3d.visualization.draw_geometries([outlier_cloud])
        # Visualize the results
        o3d.visualization.draw_geometries([inlier_cloud])
        #################################################
        ################# get tsdf and occu #####################
        Nx, Ny = 50,50
        #################################
        ''' pico '''
        x = np.linspace(-0.42, 0.08, Nx)
        y = np.linspace(-0.34, 0.16, Ny)
        
        '''realsense'''
        '''
        x = np.linspace(-0.2, 0.3, Nx)
        y = np.linspace(-0.34, 0.16, Ny)
        '''
        ################################
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((Nx*Ny,3))
        grid[:,0] = xv.flatten()
        grid[:,1] = yv.flatten()
        pts_grid = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid))
        outlier_cloud_np[:,2] = 0
        outlier_cloud_tmp = o3d.geometry.PointCloud()
        outlier_cloud_tmp.points = o3d.utility.Vector3dVector(outlier_cloud_np)
        distance = pts_grid.compute_point_cloud_distance(outlier_cloud_tmp)
        dist = np.array(distance)
        Tsdf = dist.reshape(Ny,Nx)
        Tsdf = Tsdf*255
        Tsdf=Tsdf.astype(np.uint8)
        #Tsdf = np.flipud(Tsdf)
        image_name = 'tsdf2' + ".png"
        print(image_name)
        cv2.imwrite(path+image_name,Tsdf)
        pts = outlier_cloud_np.copy()
        ##############################
        ''' pico'''
        u = (pts[:,0] +0.42)/ ( 0.5 )
        v = (pts[:,1] +0.34)/ ( 0.5 )
        
        '''realsense'''
        '''
        u = (pts[:,0] +0.2)/ ( 0.5 )
        v = (pts[:,1] +0.34)/ ( 0.5 )
        '''
        ##############################
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
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(occupancy,-1,kernel)
        img_blur=np.where(img_blur <100, 0, img_blur)
        img_blur=np.where(img_blur >=100, 255, img_blur)
        #occupancy = np.flipud(occupancy)
        occupancy=occupancy.astype(np.uint8)
        tmp_data = np.zeros([1,50,50,2])
        tmp_data[0,:,:,0] = occupancy.copy()
        tmp_data[0,:,:,1] = Tsdf.copy()
        fileObject = open('./data/tmp_data.pkl', 'wb')

        pickle.dump(tmp_data, fileObject)
        fileObject.close()
        image_name = 'occupancy2' + ".png"
        print(image_name)
        cv2.imwrite(path+image_name,occupancy)
        cv2.imwrite(path+'image_blur.png',img_blur)
        #################################################
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
        height_map = np.zeros((50, 50))
        proj_points += np.abs(proj_points.min(axis=0))
        proj_points *= 50 / proj_points.max(axis=0)
        proj_points = proj_points.astype(int)
        proj_points[proj_points < 0] = 0
        proj_points[proj_points >= 50] = 49
        print(np.max(points[:, 2]),np.min(points[:, 2]))
        points[:, 2] = (points[:, 2]-np.min(points[:, 2]))*255/(np.max(points[:, 2])-np.min(points[:, 2]))
        height_map[proj_points[:, 0], proj_points[:, 1]] = points[:, 2]
        height_map.astype(np.uint8)
        image_name = 'heightmap'+str(i) + ".png"
        print(image_name)
        cv2.imwrite(path+image_name,height_map)
        if i > 11:
            break
