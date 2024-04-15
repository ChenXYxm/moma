import rospy
from geometry_msgs.msg import PoseStamped

def callback(data):
    # This callback function will be called whenever a new message is received on the topic
    rospy.loginfo("Received a new message from /cartesian_impedance_example_controller/equilibrium_pose")

    # Modify the received message, for example, change the position
    data.pose.position.x = 0.5
    data.pose.position.y = 0.25
    data.pose.position.z = 0.16

    # Publish the modified message back to the same topic
    pub.publish(data)

def listener():
    # Initialize the ROS node
    rospy.init_node('custom_publisher', anonymous=True)

    # Subscribe to the topic
    rospy.Subscriber("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, callback)

    # Create a publisher for the same topic
    global pub
    pub = rospy.Publisher("/cartesian_impedance_example_controller/equilibrium_pose", PoseStamped, queue_size=10)

    # Spin to keep the script alive
    rospy.spin()

if __name__ == '__main__':
    listener()
