#!/usr/bin/env python

#!/usr/bin/env python

import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarker, Marker, InteractiveMarkerControl
from geometry_msgs.msg import Pose

def change_marker_position(x, y, z):
    # Initialize the ROS node
    rospy.init_node('change_marker_position', anonymous=True)

    # Create a publisher for the /interactive_marker_server/update topic
    marker_pub = rospy.Publisher('/equilibrium_pose_marker/update', InteractiveMarkerUpdate, queue_size=1)

    # Create an InteractiveMarkerUpdate message
    update_msg = InteractiveMarkerUpdate()

    # Create an InteractiveMarker
    interactive_marker = InteractiveMarker()
    interactive_marker.header.frame_id = "panda_link0"  # Replace with the appropriate frame ID
    interactive_marker.name = "equilibrium_pose"  # Make sure it matches the marker's name

    # Create a Pose for the new position
    new_pose = Pose()
    new_pose.position.x = x
    new_pose.position.y = y
    new_pose.position.z = z

    # Set the new pose in the InteractiveMarker
    interactive_marker.pose = new_pose

    # Create an InteractiveMarkerControl (you might need to adapt this based on your specific use case)
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(Marker())  # You might need to customize this Marker based on your needs

    # Add the InteractiveMarkerControl to the InteractiveMarker
    interactive_marker.controls.append(control)

    # Add the InteractiveMarker to the update message
    update_msg.markers.append(interactive_marker)

    # Publish the InteractiveMarkerUpdate
    marker_pub.publish(update_msg)

if __name__ == '__main__':
    try:
        # Specify the new position values
        new_x = 1.0
        new_y = 2.0
        new_z = 0.5

        # Call the function to change the marker position
        change_marker_position(new_x, new_y, new_z)

    except rospy.ROSInterruptException:
        pass

'''
import rospy
from visualization_msgs.msg import InteractiveMarkerUpdate, InteractiveMarker, Marker, InteractiveMarkerControl
from geometry_msgs.msg import Pose, Point
from geometry_msgs.msg import PoseStamped
def change_marker_position(x, y, z):
    # Initialize the ROS node
    rospy.init_node('change_marker_position', anonymous=True)

    # Create a publisher for the /interactive_marker/update topic
    marker_pub = rospy.Publisher('/equilibrium_pose_marker/update', InteractiveMarkerUpdate, queue_size=10)
    
    # Create an InteractiveMarkerUpdate message
    update_msg = InteractiveMarkerUpdate()

    # Create an InteractiveMarker
    interactive_marker = InteractiveMarker()
    interactive_marker.header.frame_id = "panda_link0"  # Replace with the appropriate frame ID
    interactive_marker.name = "equilibrium_pose"  # Make sure it matches the marker's name

    # Create a Pose for the new position
    new_pose = Pose()
    new_pose.position.x = x
    new_pose.position.y = y
    new_pose.position.z = z

    # Set the new pose in the InteractiveMarker
    interactive_marker.pose = new_pose

    # Create an InteractiveMarkerControl (you might need to adapt this based on your specific use case)
    control = InteractiveMarkerControl()
    control.always_visible = True
    control.markers.append(Marker())  # You might need to customize this Marker based on your needs

    # Add the InteractiveMarkerControl to the InteractiveMarker
    interactive_marker.controls.append(control)

    # Set the InteractiveMarker in the InteractiveMarkerUpdate
    update_msg.server_id = "/interactive_marker"  # Replace with the correct server ID
    update_msg.changes.append(InteractiveMarkerUpdate.POSE_UPDATE)
    update_msg.pose_update.append(interactive_marker)
    
    # Publish the InteractiveMarkerUpdate
    marker_pub.publish(update_msg)
    
    
    

if __name__ == '__main__':
    try:
        # Specify the new position values
        new_x = 1.0
        new_y = 2.0
        new_z = 0.5

        # Call the function to change the marker position
        change_marker_position(new_x, new_y, new_z)

    except rospy.ROSInterruptException:
        pass
'''
