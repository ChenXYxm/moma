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
#from pushing.place_new_obj import place_new_obj_fun
from shapely import Polygon, STRtree, area, contains, buffer, intersection,get_coordinates
#import matplotlib.pyplot as plt
#import tkinter as tk
#########################################################
#########################################################





def place_new_obj_fun(occu_ori,new_obj):
    ######################## input format
    # occu_ori: numpy 2d array: binary 0,1
    # new_obj: numpy 2d array: vertices of bbox in relative to the center point
    ########################
    shape_occu = occu_ori.shape
    Nx = shape_occu[1]
    Ny = shape_occu[0]
    # print(shape_occu)
    num_check_edge = 0
    occu = occu_ori.copy()
    # plt.imshow(occu)
    # plt.show()
    bbox = []
    for i in range(2):
        bbox.append(np.linalg.norm(new_obj[i]-new_obj[i+1]))
    num_grid_l = int(np.ceil(np.max(np.array(bbox))))
    num_grid_s = int(np.ceil(np.min(np.array(bbox))))
    ################# change the occu to image form of the cv2
    occu = np.array(occu*255,dtype=np.uint8)
    ret,thresh = cv2.threshold(occu,50,255,0)
    ################# detect contours
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours detected:",len(contours))
    shape_dict = dict()
    i = 0
    polygons = []
    ################# build polygon for the table
    table_polygon = Polygon(np.array([[0,0],[Nx-1,0],[Nx-1,Ny-1],[0,Ny-1]]))
    ################# use polygons to approximate the contours and store the part that is inside the table
    for cnt in contours:
        i += 1
        # approx = cv2.minAreaRect(cnt)
        # box = cv2.boxPoints(approx)
        # approx = np.int0(box)
        hull = cv2.convexHull(cnt,returnPoints = True)
        # print(hull)
        approx = np.int0(hull)
        if len(approx) >=3:
            # img = cv2.drawContours(img,[approx],-1,(0,255,255),3)

            # print(approx)
            approx = approx.reshape((-1,2))
            # print(approx)
            shape_dict[i] = approx
            polygon_tmp = Polygon(approx)
            polygon_tmp_2 = table_polygon.intersection(polygon_tmp)
            if polygon_tmp_2.area>0:
                polygons.append(polygon_tmp_2)
    ################## remove wrongly detected polygons
    if len(polygons)>=1:
        tree_ori = STRtree(polygons)
        del_ind = []
        for i,poly in enumerate(polygons):
            poly_tmp = buffer(poly,distance=1)
            # print(poly_tmp,poly)
            if i == 0:
                polygons_tmp = polygons[1:]
            elif i == len(polygons)-1:
                polygons_tmp = polygons[:-1]
            else:
                polygons_tmp = polygons[:i] +polygons[i+1:]
            tree = STRtree(polygons_tmp)
            indice = tree.query(poly_tmp, predicate="contains").tolist()
            if len(indice) >0:
                for j in indice:
                    if contains(poly_tmp,tree.geometries.take(j)) and area(poly_tmp)>area(tree.geometries.take(j)):
                        if j >= i:
                            j_tmp = j+1
                        else:
                            j_tmp = j
                        if j_tmp+1 in shape_dict:
                            # print("show remove")
                            # print(j_tmp)
                            del(shape_dict[j_tmp+1])
                            del_ind.append(int(j_tmp))
        shape_dict = dict()
        polygons_tmp = polygons.copy()
        # print(del_ind)
        polygons = []
        for i in range(len(polygons_tmp)):
            if i not in del_ind:
                polygons.append(polygons_tmp[i])

        occu_tmp = occu_ori.copy()
        for i in range(len(polygons)):
            # print(polygons[i])
            
            # np.asarray(polygons[i])
            # print(polygons[i].coords)
            # poly_points = np.array(polygons[i]).reshape(-1)
            # print(poly_points)
            # print(get_coordinates(polygons[i]))
            poly_points_tmp = np.array(np.round(np.array(get_coordinates(polygons[i]))),dtype=np.int8)
            shape_dict[i] = poly_points_tmp.copy()
            # print('points of polygons')
            # print(poly_points_tmp)
            # print(shape_dict)
            for j in range(len(shape_dict[i])):
                ## TODO
                if shape_dict[i][j][1] >= occu_tmp.shape[0]:
                    shape_dict[i][j][1] = occu_tmp.shape[0]-1
                if shape_dict[i][j][0] >= occu_tmp.shape[1]:
                    shape_dict[i][j][0] = occu_tmp.shape[1]-1
                if shape_dict[i][j][1] < 0:
                    shape_dict[i][j][1] = 0
                if shape_dict[i][j][0] < 0:
                    shape_dict[i][j][0] = 0    
                occu_tmp[shape_dict[i][j][1],shape_dict[i][j][0]] = 3
        # print('shape mask')
        # # plt.imshow(occu_tmp)
        # # plt.show()
        # print(shape_dict)
        length_dict = dict()
        length_list = []
        # occu_tmp = np.array(occu).copy()
        for i in shape_dict:
            points_tmp = shape_dict[i].copy()
            points_tmp[:,0] = points_tmp[:,1]
            points_tmp[:,1] = shape_dict[i][:,0].copy()
            for j in range(len(points_tmp)-1):
                p_s = points_tmp[j]
                # if j < len(points_tmp)-1:
                #     p_e = points_tmp[j+1]
                # else:
                #     p_e = points_tmp[0]
                p_e = points_tmp[j+1]
                line = p_e - p_s
                length = np.linalg.norm(line)
                if length not in length_dict:
                    length_dict[length] = [p_s,p_e]
                    length_list.append(length)
                else:
                    length_dict[length].append(p_s)
                    length_dict[length].append(p_e)
                # print(p_s,p_e,length)
                for k in range(int(np.ceil(length))):
                    tmp_delta = [k*line[0]/length,k*line[1]/length]
                    for _,l in enumerate(tmp_delta):
                        if l >=0:
                            tmp_delta[_] = np.ceil(l)
                        else:
                            tmp_delta[_] = np.floor(l)
                    occu_tmp[int(np.round(p_s[0]+tmp_delta[0])),int(np.round(p_s[1]+tmp_delta[1]))] = 3
                    # occu_tmp[int(np.round(p_s[0]+k*line[0]/length)),int(np.round(p_s[1]+k*line[1]/length))] = 2
        length_list.append(Ny)
        length_dict[Ny] = [np.array([0,0]),np.array([Ny-1,0])]
        length_dict[Ny].append(np.array([0,Nx-1]))
        length_dict[Ny].append(np.array([Ny-1,Nx-1]))
        length_list.append(Nx)
        length_dict[Nx] = [np.array([0,0]),np.array([0,Nx-1])]
        length_dict[Nx].append(np.array([Ny-1,0]))
        length_dict[Nx].append(np.array([Ny-1,Nx-1]))
        # occu_tmp[Ny-1,Nx-1] = 3
        # plt.imshow(occu_tmp)
        # plt.show()
        occu_tmp2 = occu_tmp.copy()
        flag_found = False
        dila_polygons = []
        for i in polygons:
            dila_polygons.append(i.buffer(2))
        tree = STRtree(dila_polygons)
        # print("new obj shape")
        # print([num_grid_l,num_grid_s])
        for length_ori in [num_grid_l,num_grid_s]:
            if length_ori == num_grid_l:
                length_other = num_grid_s
            else:
                length_other = num_grid_l

            length_arr = abs(np.array(length_list)-length_ori)
            for i in range(len(length_list)):
                # print(i)
                # if flag_found:
                #     break
                ind_tmp = np.argmin(length_arr)
                for m in range(int(len(length_dict[length_list[ind_tmp]])/2)):
                    num_check_edge +=1
                    # print("num check edge")
                    # print(num_check_edge,len(length_dict[length_list[ind_tmp]]))
                    p_s_first = length_dict[length_list[ind_tmp]][2*m]
                    p_e_first = length_dict[length_list[ind_tmp]][2*m+1]
                    p_s_first = [p_s_first[1],p_s_first[0]]
                    p_e_first = [p_e_first[1],p_e_first[0]]
                    # print("points")
                    # print(p_s,p_e)
                    line = np.array(p_e_first) - np.array(p_s_first)
                    line = line.astype(np.float16)
                    length = np.linalg.norm(line)
                    offset_l = [0.,1.,2.,3.,4.,5.,6.,7.,8.,-1.,-2.,-3.,-4.,-5.,-6,-7.,-8.]
                    if length < 1:
                        length_arr[ind_tmp] = 1000
                        continue
                    for p in offset_l:
                        # p_s = length_dict[length_list[ind_tmp]][2*m]
                        # p_e = length_dict[length_list[ind_tmp]][2*m+1]
                        # p_s = [p_s[1],p_s[0]]
                        # p_e = [p_e[1],p_e[0]]
                        # # print("points")
                        # # print(p_s,p_e)
                        # line = np.array(p_e) - np.array(p_s)
                        # length = np.linalg.norm(line)
                        delta_l = length_ori-length
                        p_s_ori = np.array(p_s_first).copy() + p*line.copy()/length
                        p_e_ori = (np.array(p_e_first).copy() + delta_l*line.copy()/length).copy() + p*line.copy()/length
                        # print(p_s_ori,p_e_ori,line,length_ori,length,delta_l,p_s_first,p_e_first)
                        if p == 0:
                            # print("ordinary")
                            range_o = int(np.ceil(abs(delta_l)))
                        else:
                            range_o = 1
                        for o in range(range_o):
                            p_s = p_s_ori - np.sign(delta_l)*o*line.copy()/length
                            p_e = p_e_ori - np.sign(delta_l)*o*line.copy()/length
                            # print("check original points")
                            # print(p_s,p_e,p,o)
                            for gap in range(2,6):
                                for n in range(2):
                                    sign = (-1)**n
                                    tmp_delta = gap*line.copy()/length
                                    # print("s points")
                                    # print(tmp_delta,gap,line,length)
                                    # print(length_other,length_ori)
                                    tmp_delta = np.array([(tmp_delta[1]*sign),(tmp_delta[0]*sign*(-1))])
                                    p_s_new = p_s + tmp_delta
                                    p_e_new = p_e + tmp_delta
                                    p_s_next = tmp_delta*length_other/gap + p_s_new
                                    # p_s_next = np.array(p_s_next,dtype=int)
                                    p_e_next = tmp_delta*length_other/gap + p_e_new
                                    # p_e_next = np.array(p_e_next,dtype=int)
                                    # bound_box = Polygon([[0,0],[0,60],[100,0],[100,60]])
                                    # print(bound_box)
                                    # print(gap,tmp_delta,length_other)
                                    new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                    new_poly_vetices = np.array(new_poly_vetices,dtype=np.uint8).reshape((-1,2))
                                    if (np.max(new_poly_vetices[:,0])< Nx-1 and np.max(new_poly_vetices[:,1])< Ny-1
                                        and np.min(new_poly_vetices[:,0])>=1 and np.min(new_poly_vetices[:,1])>=1):
                                        new_poly_vetices = [p_s_new,p_e_new,p_e_next,p_s_next]
                                        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
                                        points_tmp = new_poly_vetices.copy()
                                        points_tmp[:,1] = points_tmp[:,0].copy()
                                        points_tmp[:,0] = new_poly_vetices[:,1].copy()
                                        poly = Polygon([p_s_new,p_e_new,p_e_next,p_s_next])
                                        if not poly.is_valid:
                                            continue
                                        indices = tree.nearest(poly)
                                        nearest_poly = tree.geometries.take(indices)
                                        indices_2 = tree.query(poly)

                                        if nearest_poly.is_valid and poly.disjoint(nearest_poly) and area(intersection(poly,nearest_poly))==0 and len(indices_2)==0:

                                            for j in range(len(points_tmp)):
                                                p_s_1 = points_tmp[j]
                                                if j < len(points_tmp)-1:
                                                    p_e_1 = points_tmp[j+1]
                                                else:
                                                    p_e_1 = points_tmp[0]
                                                line_1 = p_e_1 - p_s_1
                                                length_1 = np.linalg.norm(line_1)
                                                for k in range(int(np.ceil(length_1))):
                                                    tmp_delta_1 = [k*line_1[0]/length_1,k*line_1[1]/length_1]
                                                    for _,l in enumerate(tmp_delta_1):
                                                        if l >=0:
                                                            tmp_delta_1[_] = np.ceil(l)
                                                        else:
                                                            tmp_delta_1[_] = np.floor(l)
                                                    if np.round(p_s_1[0]+tmp_delta_1[0])>=Ny:
                                                        tmp_delta_1[0]=Ny-1-p_s_1[0]
                                                    if np.round(p_s_1[1]+tmp_delta_1[1])>=Nx:
                                                        tmp_delta_1[1] = Nx-1 - p_s_1[1]
                                                    occu_tmp[int(np.round(p_s_1[0]+tmp_delta_1[0])),int(np.round(p_s_1[1]+tmp_delta_1[1]))] = 3
                                            flag_found = True
                                            new_obj_pos = get_pos(new_obj,new_poly_vetices)
                                            current_time = rospy.Time.now().to_sec()
                                            current_time = np.round(current_time)
                                            current_time = int(current_time)
                                            cv2.imwrite('./data/point_cloud/placing_result/'+str(current_time)+'.png',occu_tmp*255/3)
                                            print('saved plcaing result image')
     
                                            return flag_found,new_poly_vetices,occu_tmp,new_obj_pos
                                            break
                                    if flag_found:
                                        break
                                if flag_found:
                                    break
                            if flag_found:
                                break
                        if flag_found:
                            break
                    if flag_found:
                        break
                length_arr[ind_tmp] = 1000
        return False,None, None,None
    else:
        occu_tmp = np.array(occu)
        l = int(np.ceil(bbox[0]))
        w = int(np.ceil(bbox[1]))
        new_poly_vetices = [[0,Ny-w-1],[l,Ny-1-w],[l,Ny-1],[0,Ny-1]]
        new_poly_vetices = np.array(new_poly_vetices).reshape((-1,2))
        for j in range(len(new_poly_vetices)):
            occu_tmp[new_poly_vetices[j][1],new_poly_vetices[j][0]] = 3
        new_obj_pos = get_pos(new_obj,new_poly_vetices)
        return True,new_poly_vetices,occu_tmp,new_obj_pos
def get_pos(new_obj,new_poly_vetices):
    l1 = []
    l2 = []
    pos = [0,0,0]
    obj_l1 = np.array(new_obj[1]) - np.array(new_obj[0])
    obj_l2 = np.array(new_obj[3]) - np.array(new_obj[0])
    sign_obj_p0 = np.sign(np.cross(obj_l1,obj_l2))
    for i in range(2):
        l1.append(np.linalg.norm(new_obj[i]-new_obj[i+1]))
        l2.append(np.linalg.norm(new_poly_vetices[i]-new_poly_vetices[i+1]))
    if l2[0] <1:
        l2[0] = 1
    if l2[1] < 1:
        l2[1] = 1
    if l1[1] < 1:
        l1[1] = 1
    if l1[0] < 1:
        l1[0] = 1
    # print(l1,l2)
    if l1[0] >=l1[1]:
        if l2[0]>=l2[1]:
            
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
    else:
        if l2[0]<l2[1]:
            l_tmp = new_poly_vetices[1]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[3]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[0][0])*l_tmp/l2[0] + abs(new_obj[0][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[2][0])*l_tmp/l2[0] + abs(new_obj[2][1])*l_tmp_2/l2[1]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
        else:
            l_tmp = new_poly_vetices[3]-new_poly_vetices[0]
            l_tmp_2 = new_poly_vetices[1]-new_poly_vetices[0]
            sign_pos_p0 = np.sign(np.cross(l_tmp,l_tmp_2))
            
            if sign_obj_p0 == sign_pos_p0:
                angle = np.arctan2(-l_tmp[1],-l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[1][0])*l_tmp/l2[1] + abs(new_obj[1][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]
            else:
                
                angle = np.arctan2(l_tmp[1],l_tmp[0])
                pos_tmp = new_poly_vetices[0] + abs(new_obj[3][0])*l_tmp/l2[1] + abs(new_obj[3][1])*l_tmp_2/l2[0]
                pos[2] = angle
                pos[0] = pos_tmp[1]
                pos[1] = pos_tmp[0]

    return pos


##########################################################
##########################################################
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
    flag_get = True
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
    
    
    if occu[actions.flatten()[0],actions.flatten()[1]] >0:
       flag_get = False
    
    print('value: ', value)
    occu = occu - 100
    occu[actions.flatten()[0],actions.flatten()[1]] = 255
    current_time = rospy.Time.now().to_sec()
    current_time = np.round(current_time)
    current_time = int(current_time)
    image_name = "./data/point_cloud/pushing_result/"+str(current_time)+".png"
    cv2.imwrite(image_name,occu)
    '''
    for _ in range(len(value)):
            if float(value[_]) <=-0.1:
                act_app[_] = 10
    '''
    actions_new = np.c_[actions,act_app.T]   
    return actions_new,flag_get


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
    def go_to_obs_joint_state(self):

        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = -tau / 10
        joint_goal[2] = 0
        joint_goal[3] = -tau / 4
        joint_goal[4] = 0
        joint_goal[5] = tau / 5.2  # 1/6 of a turn
        joint_goal[6] = tau / 8
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        current_joints = move_group.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01)
    def go_to_get_new_obj_pose(self):
        move_group = self.move_group
        move_group.set_max_velocity_scaling_factor(0.5)
        joint_goal = move_group.get_current_joint_values()
        joint_goal[0] = 0
        joint_goal[1] = 0
        joint_goal[2] = -tau / 9.5
        joint_goal[3] = -tau/4
        joint_goal[4] = 0
        joint_goal[5] = tau/4  # 1/6 of a turn
        joint_goal[6] = 0
        move_group.go(joint_goal, wait=True)
        move_group.stop()
        
        ################# adjust the position
        current_joints = move_group.get_current_joint_values()
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.w = 0.0
        pose_goal.orientation.x = 1.0
        pose_goal.orientation.y = 0.0
        pose_goal.position.x = 0.4  ### 0.74 - 0.32
        pose_goal.position.y = -0.35
        pose_goal.position.z = 0.55
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        # Calling `stop()` ensures that there is no residual movement
        move_group.stop()
        # It is always good to clear your targets after planning with poses.
        # Note: there is no equivalent function for clear_joint_value_targets().
        move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        #current_joints = move_group.get_current_joint_values()
        #print('current_joints: ',current_joints)
        
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
        pose_goal.position.z = 0.3
        
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
            print('cannot find path to move the gripper to the target point: going down1')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.20
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down2')
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.172
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down3')
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
    def place(self,x=0.5,y = 0.0,rotate=0,z=0.185):
       
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
        if joint_goal[6]+rotate<2.8 and joint_goal[6]+rotate>-2.8:
            joint_goal[6] = joint_goal[6]+rotate
        else:
            joint_goal[6] = joint_goal[6]-tau/2.0+rotate 
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
        pose_goal.position.z = z
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        joint_goal = move_group.get_current_joint_values()
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
        return all_close(pose_goal, current_pose, 0.05)
    def move_up(self):
        move_group = self.move_group
        pose_goal = move_group.get_current_pose().pose
        pose_goal.position.z = 0.23
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        print('success: ', success)
        
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper up after placing')
            
    def grasp(self,x=0.45,y = -0.35,rotate=0,z=0.16):
       
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
        pose_goal.position.z = 0.3
        
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
        
        ############################ go down ####################################
        pose_goal = self.move_group.get_current_pose().pose
        #helper_frame_pose.header.frame_id = "world"
        pose_goal.position.z = 0.24
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
        pose_goal.position.z = z
        move_group.set_pose_target(pose_goal)
        success = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()
        current_pose = self.move_group.get_current_pose().pose
        
        if not all_close(pose_goal, current_pose, 0.05):
            print('cannot find path to move the gripper to the target point: going down')
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
        
        # Subscribe to the depth image topic
        #self.puhsing_controller.go_to_obs_joint_state()
        self.puhsing_controller.go_to_get_new_obj_pose()
        self.gripper.release(width=0.12)
        self.gripper.grasp()
        rospy.Subscriber('/wrist_camera/depth/camera_info', CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber('/wrist_camera/depth/image_rect_raw', Image, self.depth_image_callback, queue_size=1,buff_size=2**30) ##307200
        #self.debug_pub = rospy.Publisher('/debug_depth', Image, queue_size=1)
        self.time = 0.0
        self.flag_place = True
        self.pre_image = np.zeros([50,50]) 
        #rospy.spin()
    def quaternion_to_matrix(self,q):
        x, y, z,w  = q
        self.R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2]
        ])
        # return self.R
    def depth_pixel_to_camera_obs(self,cv_image):
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        points_new_obj = points.copy()
        #o3d.visualization.draw_geometries([pcd])
        '''
        ######################################## drawing
        points = points[np.where(points[:,0]<0.43)]
        points = points[np.where(points[:,0]>-0.3)]
        points = points[np.where(points[:,1]<0.22)]
        points = points[np.where(points[:,1]>-0.32)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot wrist camera##################
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        points = points[np.where(points[:,0]<0.28)]
        points = points[np.where(points[:,0]>-0.235)]
        points = points[np.where(points[:,1]<0.207)]
        points = points[np.where(points[:,1]>-0.290)]
        ########################################
        
        
        
        ##########################################
        ## TODO: switch between real and simulation
        ############# for simulation #############
        '''
        points = points[np.where(points[:,0]<0.327)]
        points = points[np.where(points[:,0]>-0.173)]
        points = points[np.where(points[:,1]<0.223)]
        points = points[np.where(points[:,1]>-0.277)]
        '''
        ##########################################
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ##########################################
        ###### get the shape info of new obj #####
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]<-0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,0]>-0.38)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]<0.15)]
        points_new_obj = points_new_obj[np.where(points_new_obj[:,1]>-0.290)]
        ##########################################
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_new_obj[:,:3])
        o3d.visualization.draw_geometries([pcd_obj])
        ##########################################
        '''
        return pcd, points
    def depth_pixel_to_camera_obj(self,cv_image):
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
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        
        #o3d.visualization.draw_geometries([pcd])
        '''
        ######################################## drawing
        points = points[np.where(points[:,0]<0.43)]
        points = points[np.where(points[:,0]>-0.3)]
        points = points[np.where(points[:,1]<0.22)]
        points = points[np.where(points[:,1]>-0.32)]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        '''
        ########################################
        ## TODO: switch between real and simulation
        ########## real robot wrist camera##################
        #points = points[np.where(points[:,2]<1)]
        #print(np.max(points[:,0]),np.min(points[:,0]),np.max(points[:,1]),np.min(points[:,1]),np.max(points[:,2]),np.min(points[:,2]))
        points = points[np.where(points[:,0]<0.25)]
        points = points[np.where(points[:,0]>-0.04)]
        points = points[np.where(points[:,1]<0.18)]
        points = points[np.where(points[:,1]>-0.18)]
        ########################################
        
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        o3d.visualization.draw_geometries([pcd])
        #o3d.io.write_point_cloud(self.save_path+"pcd_camera_frame.ply", pcd)
        return pcd
    def image_processing(self,image):
        kernel = np.ones((3,3),np.float32)/9
        img_blur = cv2.filter2D(image,-1,kernel)
        img_blur=np.where(img_blur <100, 0, img_blur)
        img_blur=np.where(img_blur >=100, 255, img_blur)
        cv2.imwrite(self.save_path+'occu_filtered_new.png',img_blur)
        current_time = rospy.Time.now().to_sec()
        current_time = np.round(current_time)
        current_time = int(current_time)
        cv2.imwrite(self.save_path+'occu/'+str(current_time)+'.png',img_blur)
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
        print('world: ',np.max(world_points[:,0]),np.min(world_points[:,0]),np.max(world_points[:,1]),np.min(world_points[:,1]))
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
        #o3d.io.write_point_cloud(self.save_path+"pcd_world2.ply", pcd)
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        self.pcd = pcd
        self.inlier_cloud = pcd.select_by_index(inliers)
        # Extract outliers (non-plane points)
        self.outlier_cloud = pcd.select_by_index(inliers, invert=True)
        # Visualize the results
        print('table pcd')
        o3d.visualization.draw_geometries([self.inlier_cloud])
        o3d.visualization.draw_geometries([self.outlier_cloud])
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
        o3d.visualization.draw_geometries([outlier_cloud])
        self.outlier_cloud = outlier_cloud
        # Visualize the results
        
        return pcd
    def get_center_new_obj(self,pcd_obj):
        points_np = np.asarray(pcd_obj.points)
        for i in range(len(points_np)):
            points_np[i] = np.dot(self.R,points_np[i])+self.trans
        pcd_obj = o3d.geometry.PointCloud()
        pcd_obj.points = o3d.utility.Vector3dVector(points_np)
        plane_model, inliers = pcd_obj.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=1000)
        obj_outlier_cloud = pcd_obj.select_by_index(inliers)
  
        obj_inlier_cloud = pcd_obj.select_by_index(inliers, invert=True)
        #o3d.visualization.draw_geometries([obj_inlier_cloud])
        o3d.visualization.draw_geometries([obj_outlier_cloud])
        obj_inlier_cloud_np = np.array(obj_inlier_cloud.points)
        obj_inlier_cloud_list = []
        ############################################
        A,B,C,D = plane_model
        for i in range(len(obj_inlier_cloud_np)):
            x,y,z = obj_inlier_cloud_np[i]
            numerator = A * x + B * y + C * z + D
            distance = np.abs(numerator)/np.sqrt(A**2 + B**2 + C**2)
            if numerator > 0 and distance<0.3:
                obj_inlier_cloud_list.append(obj_inlier_cloud_np[i])
        obj_inlier_cloud = o3d.geometry.PointCloud()    
        obj_inlier_cloud_np = np.array(obj_inlier_cloud_list).reshape(-1,3) 
        #obj_inlier_cloud_np[:,2] = 0 
        obj_inlier_cloud.points = o3d.utility.Vector3dVector(obj_inlier_cloud_np)
        height = np.mean(obj_inlier_cloud_np[:,2]) #- np.mean(np.array(obj_outlier_cloud.points)[:,2])
        #print('point cloud center')
        aabb = obj_inlier_cloud.get_axis_aligned_bounding_box()
        aabb.color = [0,0,1]
        #o3d.visualization.draw_geometries([obj_inlier_cloud,aabb])
        obj_center = obj_inlier_cloud.get_center()
        aabb_center = aabb.get_center()
        print(obj_center,aabb_center)
        print('length')
        length = aabb.get_extent()
        print(length)
        return aabb_center,length,height
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
    def postprocessed_occu_to_tsdf(self,occu):
        Nx,Ny = self.occu_size
        occu_index = np.where(occu>=0.5)
        print('occu index',occu_index)
    def pcd_to_heightmap(self,pcd_w):
        xyz = np.asarray(pcd_w.points)
        
        #print('xyz shape: ',xyz.shape)
        Nx, Ny = self.occu_size
        
        u = (xyz[:,0] - self.min_x)/ ( self.max_x-self.min_x )
        v = (xyz[:,1] - self.min_y)/ ( self.max_y-self.min_y )
        u = (Nx-1)*u
        v = (Ny-1)*v
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        heightmap = np.zeros( (Ny,Nx) )
        xyz[:,0] = u
        xyz[:,1] = v
        for i in range(50):
            for j in range(50):
                tmp_xyz = xyz[np.where(xyz[:,0]==i)].copy()
                tmp_xyz = tmp_xyz[np.where(tmp_xyz[:,1]==j)].copy()
                if len(tmp_xyz)>0:
                    heightmap[i,j] = np.mean(tmp_xyz[:,2])
        
        heightmap_tmp = heightmap.copy()
        heightmap_tmp = (heightmap_tmp-np.min(heightmap_tmp))/(np.max(heightmap_tmp)-np.min(heightmap_tmp))
        heightmap_tmp = heightmap_tmp*255
        # heightmap = heightmap.astype(np.uint8)
        cv2.imwrite(self.save_path+'heightmap.png',heightmap_tmp)
        return heightmap
        
    def decode_image_pos(self):
        pass
    def depth_image_callback(self, msg):
        msg_time = msg.header.stamp.to_sec()
        current_time = rospy.Time.now().to_sec()
        print("Delay is ", current_time - msg_time)
        if current_time - msg_time < 2.0:
            try:
                #self.debug_pub.publish(msg)
                # Transform pixel coordinates to camera coordinates
                '''
                print(msg.header.stamp)
                print(msg.header.seq)
                print(msg.header.frame_id)
                '''
                msg_time = msg.header.stamp.to_sec()
                current_time = rospy.Time.now().to_sec()
                print("Delay is ", current_time - msg_time,"current time is: ",current_time)
            
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
                print(cv_image.shape)
            
                #plt.imshow(cv_image)
                ''' debuging
                cv_image_tmp = cv_image.copy()
                cv_image_tmp[np.where(cv_image_tmp<=900)] = 0
                cv_image_tmp[np.where(cv_image_tmp>900)] = 255
                cv_image_tmp = cv_image_tmp.astype(np.uint8)
                cv2.imwrite(self.save_path+'depth.png',cv_image_tmp)
                '''
                camera_frame = "/wrist_camera_depth_optical_frame"  # Update with your camera frame
                world_frame = "/world"    # Update with your world frame

                (self.trans, self.quat) = self.listener.lookupTransform(world_frame, camera_frame, rospy.Time(0))
                #print(self.quat,self.trans)
                # get the rotation matrix
                self.quaternion_to_matrix(self.quat)
                if self.flag_place:
                
                    pcd_obj = self.depth_pixel_to_camera_obj(cv_image)
                    
                    
                    self.new_obj_center, self.new_obj_length,self.height = self.get_center_new_obj(pcd_obj)
                    w_obj = int(np.ceil(self.new_obj_length[1]*100/2.0))
                    l_obj = int(np.ceil(self.new_obj_length[0]*100/2.0))
                    print('new obj mask length: ',w_obj,l_obj,self.height)
                    self.obj_vertices = np.zeros((4,2))
                    self.obj_vertices[0,0] = -int(w_obj)
                    self.obj_vertices[0,1] = -int(l_obj)
                    self.obj_vertices[1,0] = int(w_obj)
                    self.obj_vertices[1,1] = -int(l_obj)
                    self.obj_vertices[2,0] = int(w_obj)
                    self.obj_vertices[2,1] = int(l_obj)
                    self.obj_vertices[3,0] = -int(w_obj)
                    self.obj_vertices[3,1] = int(l_obj)
                    if w_obj>2 or l_obj>2:
                        self.puhsing_controller.go_to_obs_joint_state()
                        self.flag_place = False
                # Transform camera coordinates to world coordinates using tf
                else:
                    pcd,points = self.depth_pixel_to_camera_obs(cv_image)
                    world_points = points[:,:3].copy()
                    pcd_w = self.camera_to_world(world_points)
                    occu = self.pcd_to_occu()
                    occu = self.image_processing(occu)
                    occu_tmp = occu.copy()
                    occu = occu.astype(np.uint8)
                    self.occu = occu.copy()
                    tsdf = self.pcd_to_tsdf()
                    tsdf_tmp = tsdf.copy()
                    tsdf = tsdf.astype(np.uint8)
                    cv2.imwrite(self.save_path+'occu_new.png',occu)
                    cv2.imwrite(self.save_path+'tsdf_new.png',tsdf)
                    occu = occu/255
                    pre_image = self.pre_image + occu
                    pre_image = pre_image*255/float(np.max(pre_image))
                    pre_image = pre_image.astype(np.uint8)
                    cv2.imwrite(self.save_path+'difference.png',pre_image)
                    flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,self.obj_vertices)
                    if _ is not None:
                    	self.pre_image = _.copy()
                    
                    print('pre max', np.max(self.pre_image))
                    if flag_found:
                        print('place succeed')
                        print(new_obj_pos)
                        print('vertices: ', new_poly_vetices)
                        #heightmap = self.pcd_to_heightmap(pcd_w)
                        #height = np.mean(heightmap[int(np.min(new_poly_vetices[:,0])):int(np.max(new_poly_vetices[:,0])),int(np.min(new_poly_vetices[:,1])):int(np.max(new_poly_vetices[:,1]))])
                        #print(height)
                        if self.height < 0.165:
                            print('height to low: ', self.height)
                            self.height = 0.165
                        self.gripper.release(width=0.15)
                        self.puhsing_controller.grasp(x=self.new_obj_center[0],y=self.new_obj_center[1],z=self.height-0.03)
                        self.gripper.grasp()
                        x,y,rot = new_obj_pos
                        if rot <-tau/2:
                           rot = rot + tau
                        elif rot >tau/2:
                           rot = rot -tau
                        action = [x,y,2]
                        action = np.array(action)
                        x,y,_=self.transfrom_action_world(action)
                        input( "============ Press `Enter` to continue ...") 
                        self.puhsing_controller.move_up()
                        self.puhsing_controller.place(x=x,y=y,rotate=rot,z=self.height-0.03)
                        input( "============ Press `Enter` to continue ...") 
                        self.gripper.release(width=0.15)
                        self.puhsing_controller.move_up()
                        self.puhsing_controller.go_to_get_new_obj_pose()
                        self.gripper.grasp()
                        self.flag_place = True
                    else:
                        action,flag_get=get_x_y(tsdf_tmp,occu_tmp)
	            
                        #print('action: ',action)
                        x,y,rot=self.transfrom_action_world(action)
    
                        #input( "============ Press `Enter` to continue ...") 
                        self.puhsing_controller.go_to_pose_goal(x=x,y = y,rotate=rot)
                        #rospy.sleep(3)
    
                        self.puhsing_controller.go_to_obs_joint_state()
                '''
                heightmap = self.pcd_to_heightmap(pcd_w)
                occu = self.pcd_to_occu()
                occu = self.image_processing(occu)
                occu_tmp = occu.copy()
                occu = occu.astype(np.uint8)
                #occu_ros = self.bridge.cv2_to_imgmsg(occu)

                self.occu = occu.copy()
                tsdf = self.pcd_to_tsdf()
                tsdf_tmp = tsdf.copy()
                tsdf = tsdf.astype(np.uint8)
                tsdf = cv2.cvtColor(tsdf, cv2.COLOR_GRAY2BGR)
                #tsdf_ros = self.bridge.cv2_to_imgmsg(tsdf, encoding="bgr8")
            
                
           
                # Print the transformed depth image data (replace this with your logic)
                #rospy.loginfo("Transformed Depth Image Data: %s", transformed_depth_image_data)

                # Publish the transformed depth image
                #self.image_pub.publish(tsdf_ros)
                #self.image_pub_occu.publish(occu_ros)
                
                cv2.imwrite(self.save_path+'occu_new.png',occu)
                cv2.imwrite(self.save_path+'tsdf_new.png',tsdf)
                occu = occu/255
                '''
                ################################################
                ################### TODO: ######################
                '''
                new_obj_center, new_obj_length = self.get_center_new_obj(pcd_obj)
                w_obj = int(np.ceil(new_obj_length[1]*100/2.0))
                l_obj = int(np.ceil(new_obj_length[0]*100/2.0))
                print('new obj mask length: ',w_obj,l_obj)
                obj_vertices = np.zeros((4,2))
                obj_vertices[0,0] = -int(w_obj)
                obj_vertices[0,1] = -int(l_obj)
                obj_vertices[1,0] = int(w_obj)
                obj_vertices[1,1] = -int(l_obj)
                obj_vertices[2,0] = int(w_obj)
                obj_vertices[2,1] = int(l_obj)
                obj_vertices[3,0] = -int(w_obj)
                obj_vertices[3,1] = int(l_obj)
                #print(obj_vertices)
                '''
                '''
                ################################################
                if w_obj>2 and l_obj>2:
                    
                    flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,obj_vertices)
                    if not flag_found:
                        flag_found, new_poly_vetices,_,new_obj_pos = place_new_obj_fun(occu,obj_vertices)
                    if not flag_found:
                        action,flag_get=get_x_y(tsdf_tmp,occu_tmp)
	            
                        #print('action: ',action)
                        x,y,rot=self.transfrom_action_world(action)
    
                        #input( "============ Press `Enter` to continue ...") 
                        self.puhsing_controller.go_to_pose_goal(x=x,y = y,rotate=rot)
                        #rospy.sleep(3)
    
                        self.puhsing_controller.go_to_joint_state()
                        #rospy.sleep(3)
                    else:
                        print('place succeed')
                        print(new_obj_pos)
                        print('vertices: ', new_poly_vetices)
                        height = np.mean(heightmap[int(np.min(new_poly_vetices[:,0])):int(np.max(new_poly_vetices[:,0])),int(np.min(new_poly_vetices[:,1])):int(np.max(new_poly_vetices[:,1]))])
                        print(height)
                        ########################### grasp item ##################
                        self.gripper.release(width=0.15)
                        self.puhsing_controller.grasp(x=new_obj_center[0],y=new_obj_center[1]-0.00)
                        self.gripper.grasp()
                   
                        #########################################################
                    
                        x,y,rot = new_obj_pos
                        action = [x,y,2]
                        action = np.array(action)
                        x,y,_=self.transfrom_action_world(action)
                        #input( "============ Press `Enter` to continue ...") 
                        # self.puhsing_controller.go_to_pose_goal(x=x,y = y,rotate=rot)
                        #rospy.sleep(3)
                        
                        self.puhsing_controller.go_to_joint_state()
                        self.puhsing_controller.place(x=x,y=y,rotate=rot,z=height-0.003)
                        self.gripper.release(width=0.12)
                        self.puhsing_controller.move_up()
                        self.puhsing_controller.go_to_joint_state()
                        self.gripper.grasp()
                        #rospy.sleep(3)
                    
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
        x = self.min_x+(x-0.5)*(self.max_x-self.min_x)/float(Nx) 
        y = self.min_y+(y-0.5)*(self.max_y-self.min_y)/float(Ny) 
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

