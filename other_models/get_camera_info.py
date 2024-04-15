import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np
bag_file = './data/camera_info.bag'

bag = rosbag.Bag(bag_file,'r')
bag_data = bag.read_messages()

i = 0
for topic, msg, t in bag_data:
    print(msg)
    print(topic)
    
