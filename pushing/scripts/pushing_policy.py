import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
class ImageProcessor:
    def __init__(self):
        rospy.init_node('image_processor_node', anonymous=True)

        # Create a CV bridge for converting ROS images to OpenCV images
        self.bridge = CvBridge()

        # Variable to store the latest message
        self.latest_image_msg = None

        # Subscribe to the image topic
        rospy.Subscriber('/fixed_camera/depth/image_rect_raw', Image, self.image_callback)

        # Spin to keep the node alive
        rospy.spin()

    def image_callback(self, msg):
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Perform image processing (replace this with your actual image processing code)
            #processed_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Store the latest processed image message
            self.latest_image_msg = cv_image

        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")

    def get_latest_image_msg(self):
        return self.latest_image_msg

if __name__ == '__main__':
    try:
        image_processor = ImageProcessor()

        # Example of how to access the latest processed image message outside the callback
        while not rospy.is_shutdown():
            latest_msg = image_processor.get_latest_image_msg()
            if latest_msg is not None:
            # Process the latest message as needed
                print('msg')
                print(np.max(latest_msg))
            

    except rospy.ROSInterruptException:
        pass
