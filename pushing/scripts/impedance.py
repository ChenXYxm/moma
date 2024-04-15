import rospy
from geometry_msgs.msg import PoseStamped
from moma_utils.ros.panda import PandaGripperClient
class PerformPushing_Impedance(object):
    def __init__(self):
        super(PerformPushing, self).__init__()
        pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
        rate = rospy.Rate(10)  # 10Hz
    def move(self,x = 0,y=0,z=0):
        pass
def publish_pose():
    
    pub = rospy.Publisher('/cartesian_impedance_example_controller/equilibrium_pose', PoseStamped, queue_size=10)
    i = 0
    rate = rospy.Rate(10)  # 10Hz
    #gripper = PandaGripperClient()
    while i<10:
        # Create a PoseStamped message
        pose_msg = PoseStamped()
        # Fill in the data for the PoseStamped message
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "panda_link0"  # frame ID for the pose
        pose_msg.pose.position.x = .3
        pose_msg.pose.position.y = .0
        pose_msg.pose.position.z = 0.65
        pose_msg.pose.orientation.x = 1.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z =0.0
        pose_msg.pose.orientation.w = 0.0

        # Publish the PoseStamped message
        pub.publish(pose_msg)
        i += 1
        #gripper.release(width=0.1)
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('pose_publisher', anonymous=True)
    
    try:
        publish_pose()
        
        
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass
