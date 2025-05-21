# Message to hold all detected face landmarks for a single frame
std_msgs/Header header # Standard ROS Header
msg_types/Face[] faces # List of detected faces in the current frame
