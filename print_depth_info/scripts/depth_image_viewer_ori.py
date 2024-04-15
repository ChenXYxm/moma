#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
path = './data/occu/'
class DepthImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()

        # Create a subscriber for the depth image
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_callback)

        # Create a publisher for the processed image
        self.processed_image_pub = rospy.Publisher("/camera/depth/processed_image", Image, queue_size=10)

    def depth_image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Process the depth image (add your processing steps here)
            processed_image = self.process_depth_image(depth_image)

            # Convert the processed image back to ROS Image message
            processed_image_msg = self.bridge.cv2_to_imgmsg(processed_image)

            # Publish the processed image
            self.processed_image_pub.publish(processed_image_msg)

        except Exception as e:
            print(e)

    def process_depth_image(self, depth_image):
        # Add your image processing steps here
        # For example, you can perform operations like cropping, flipping, etc.
        # Ensure the processed image has the same size and encoding as the original depth image.
        processed_image = depth_image  # Placeholder for now

        return processed_image

def main():
    rospy.init_node('depth_image_processor', anonymous=True)
    depth_processor = DepthImageProcessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
    
def depth_image_callback(msg):
    try:
        # Convert ROS Image message to OpenCV image
        bridge = CvBridge()
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #depth_image = bridge.imgmsg_to_cv2(msg, '16UC1')
        print(np.max(depth_image),np.min(depth_image),np.mean(depth_image))
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
        print(np.max(image_tmp),np.min(image_tmp),np.mean(image_tmp))
        image_tmp = image_tmp*255
        image_name = "occu.png"
        cv2.imwrite(path+image_name,image_tmp)
        
    except Exception as e:
        print(e)

def main():
    rospy.init_node('depth_image_viewer', anonymous=True)
    rospy.Subscriber("/fixed_camera/depth/image_rect_raw", Image, depth_image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

