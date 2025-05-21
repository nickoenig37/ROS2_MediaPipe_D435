import rclpy
import math
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

from msg_types.msg import PoseLandmarks

class PoseVisualizationNode(Node):
    def __init__(self):
        super().__init__('pose_visualization_node')
        
        # Declare parameter for the input pose landmarks topic
        self.declare_parameter('landmarks_topic', '/pose_tracking/landmarks')
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value

        # Declare parameter for the output visualization topic
        self.declare_parameter('visualization_topic', '/pose_tracking/visualization')
        visualization_topic = self.get_parameter('visualization_topic').get_parameter_value().string_value

        # Subscriber to the pose landmarks topic
        self.landmark_subscription = self.create_subscription(
            PoseLandmarks,
            landmarks_topic,
            self.pose_landmarks_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {landmarks_topic}")

        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(MarkerArray, visualization_topic, 10)
        self.get_logger().info(f"Publishing visualization markers on {visualization_topic}")

        # Define MediaPipe Pose connections (based on the uploaded image)
        # These are pairs of landmark IDs that should be connected by lines
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),  # Head
            (0, 4), (4, 5), (5, 6), (6, 8),  # Head
            (9, 10),                        # Mouth
            (11, 12),                       # Shoulders
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), # Left Arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), # Right Arm
            (11, 23), (12, 24),             # Torso
            (23, 24)#,                       # Hips
            # (23, 25), (25, 27), (27, 29), (27, 31), # Left Leg
            # (24, 26), (26, 28), (28, 30), (28, 32)  # Right Leg
        ]

    def pose_landmarks_callback(self, msg):
        marker_array = MarkerArray()

        # --- ADD THIS SECTION TO CLEAR ALL MARKERS ---
        # Create a marker with action DELETEALL. This will clear all markers
        # with the same namespace as the one used in the MarkerArray,
        # or all markers if no namespace is specified.
        # It's good practice to send it with the same header as the active markers.
        delete_all_marker = Marker()
        delete_all_marker.header = msg.header
        delete_all_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_all_marker)
        # --- END OF ADDED SECTION ---

        for pose_idx, pose in enumerate(msg.poses):
            # Create markers for each landmark point
            for landmark in pose.landmarks:
                if landmark.landmark_id <= 24: # Only process landmarks above hip
                    marker = Marker()
                    marker.header = msg.header
                    marker.ns = f"pose_{pose.pose_id}_landmarks" # Namespace for markers from this pose
                    marker.id = landmark.landmark_id # Unique ID for each landmark marker
                    marker.type = Marker.SPHERE # Visualize landmarks as spheres
                    marker.action = Marker.ADD
                    marker.pose.position = landmark.point
                    marker.scale.x = 0.01 # Sphere diameter
                    marker.scale.y = 0.01
                    marker.scale.z = 0.01
                    marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green color for points
                    marker.lifetime.sec = 0 # Persist marker

                    marker_array.markers.append(marker)

            # Create markers for the connections (lines)
            line_list_marker = Marker()
            line_list_marker.header = msg.header
            line_list_marker.ns = f"pose_{pose.pose_id}_connections" # Namespace for lines from this pose
            line_list_marker.id = 1000 + pose.pose_id # Unique ID for the line list marker (offset to avoid conflict)
            line_list_marker.type = Marker.LINE_LIST # Visualize connections as lines
            line_list_marker.action = Marker.ADD
            line_list_marker.scale.x = 0.01 # Line width
            line_list_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) # Blue color for lines
            line_list_marker.lifetime.sec = 0 # Persist marker

            # Create points for the lines based on connections
            for connection in self.pose_connections:
                p1_id = connection[0]
                p2_id = connection[1]

                p1_point = None
                p2_point = None

                # Find the points for the connection from the current pose's landmarks
                for landmark in pose.landmarks:
                    if landmark.landmark_id == p1_id:
                        p1_point = landmark.point
                    if landmark.landmark_id == p2_id:
                        p2_point = landmark.point

                # Only add the line if both points are available (i.e., detected with valid depth)
                # and the distance is within a reasonable range
                distance_threshold = 0.90 # 90 cm (more than enough between possible landmarks)
                distance = math.sqrt(
                    (p1_point.x - p2_point.x) ** 2 +
                    (p1_point.y - p2_point.y) ** 2 +
                    (p1_point.z - p2_point.z) ** 2
                ) if p1_point and p2_point else float('inf')


                if (p1_point is not None and p2_point is not None) and distance < distance_threshold:
                    line_list_marker.points.append(p1_point)
                    line_list_marker.points.append(p2_point)

            # Append the line list marker only if it has points
            if line_list_marker.points:
                marker_array.markers.append(line_list_marker)

        # Publish the MarkerArray
        self.marker_publisher.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PoseVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()