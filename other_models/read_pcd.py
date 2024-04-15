import open3d as o3d
path = './data/point_cloud/'

pcd = o3d.io.read_point_cloud(path+'pcd_world2.ply')
o3d.visualization.draw_geometries([pcd])
