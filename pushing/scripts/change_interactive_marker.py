#!/usr/bin/env python

import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarker, Marker, InteractiveMarkerControl,InteractiveMarkerPose
from geometry_msgs.msg import Pose, Point

def move_interactive_marker(marker_name, new_pose):
    # Initialize the ROS node
    rospy.init_node('move_interactive_marker', anonymous=True)

    # Create a publisher for the /interactive_marker/update topic
    marker_pub = rospy.Publisher('/equilibrium_pose_marker/update', InteractiveMarkerUpdate, queue_size=1)

    # Create an InteractiveMarkerUpdate message
    update_msg = InteractiveMarkerUpdate()
    update_msg.server_id = "/interactive_marker"
    update_msg.seq_num = 300
    update_msg.type = 1
    '''
    update_msg.poses.header.frame_id = "panda_link0"
    update_msg.poses.pose = new_pose
    update_msg.poses.name = marker_name
    '''
    interactive_marker = InteractiveMarkerPose()
    interactive_marker.header.frame_id = "panda_link0"  # Replace with the appropriate frame ID
    interactive_marker.name = "equilibrium_pose"
    interactive_marker.pose = new_pose
    update_msg.poses.append(interactive_marker)
    interactive_marker = InteractiveMarker()
    interactive_marker.name = marker_name
    interactive_marker.header.frame_id = "panda_link0"
    interactive_marker.pose = new_pose
    update_msg.markers.append(interactive_marker)
    print(update_msg)

    # Publish the InteractiveMarkerUpdate
    marker_pub.publish(update_msg)

if __name__ == '__main__':
    try:
        # Specify the new position values
        new_x = 1.0
        new_y = 2.0
        new_z = 0.5

        # Create a Pose for the new position
        new_pose = Pose()
        new_pose.position.x = new_x
        new_pose.position.y = new_y
        new_pose.position.z = new_z

        # Specify the existing marker's name
        existing_marker_name = "equilibrium_pose"

        # Call the function to move the existing marker
        move_interactive_marker(existing_marker_name, new_pose)

    except rospy.ROSInterruptException:
        pass

