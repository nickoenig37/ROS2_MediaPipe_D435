import rclpy
import math
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


from msg_types.msg import HandLandmarks

class HandVisualizationNode(Node):
    def __init__(self):
        super().__init__('hand_visualization_node')
        
        # Declare parameter for the input hand landmarks topic
        self.declare_parameter('landmarks_topic', '/hand_tracking/landmarks')
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value

        # Declare parameter for the output visualization topic
        self.declare_parameter('visualization_topic', '/hand_tracking/visualization')
        visualization_topic = self.get_parameter('visualization_topic').get_parameter_value().string_value

        # Subscriber to the pose landmarks topic
        self.landmark_subscription = self.create_subscription(
            HandLandmarks,
            landmarks_topic,
            self.hand_landmarks_callback,
            10
        )
        self.get_logger().info(f"Subscribed to {landmarks_topic}")

        # Publisher for visualization markers
        self.marker_publisher = self.create_publisher(MarkerArray, visualization_topic, 10)
        self.get_logger().info(f"Publishing visualization markers on {visualization_topic}")

        # Define MediaPipe Pose connections (based on the uploaded image)
        # These are pairs of landmark IDs that should be connected by lines
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # wrist-thumbtip
            (0, 5), (5, 6), (6, 7), (7, 8),  # wrist-index
            (0, 9), (9, 10), (10, 11), (11, 12), # wrist-middle
            (0, 13), (13, 14), (14, 15), (15, 16), # wrist-ring
            (0, 17), (17, 18), (18, 19), (19, 20) # wrist-pinky
        ]

    def hand_landmarks_callback(self, msg):
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

        for hand_idx, hand in enumerate(msg.hands): # Iterate through each detected hand
            current_hand_id = hand.hand_id # id 0 = Left, id 1 = Right

            # Create markers for each landmark point
            for landmark in hand.landmarks:
                marker = Marker()
                marker.header = msg.header
                marker.ns = f"hand_{current_hand_id}_landmarks" # Namespace for markers from this pose
                marker.id = landmark.landmark_id # Unique ID for each landmark marker
                marker.type = Marker.SPHERE # Visualize landmarks as spheres
                marker.action = Marker.ADD
                marker.pose.position = landmark.point
                marker.scale.x = 0.009 # Sphere diameter
                marker.scale.y = 0.009
                marker.scale.z = 0.009
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                marker.lifetime.sec = 0 # Persist marker

                marker_array.markers.append(marker)

            # Create markers for the connections (lines) of the current hand
            line_list_marker = Marker()
            line_list_marker.header = msg.header
            line_list_marker.ns = f"hand_{current_hand_id}_connections" # Unique namespace per hand for lines
            line_list_marker.id = 1000 + current_hand_id # Unique ID for the line list marker (offset to avoid conflict with landmark IDs)
            line_list_marker.type = Marker.LINE_LIST # Visualize connections as lines
            line_list_marker.action = Marker.ADD
            line_list_marker.scale.x = 0.005 # Line width
            line_list_marker.color = ColorRGBA(r=0.3, g=0.1, b=1.0, a=1.0) 
            line_list_marker.lifetime.sec = 0 # Persist marker

            # Create points for the lines based on MediaPipe Hand connections
            # We need to map landmark IDs to their Point objects efficiently
            # Create a dictionary for quick lookup of points by ID
            landmark_points = {lm.landmark_id: lm.point for lm in hand.landmarks}

            # Create points for the lines based on connections
            for connection in self.hand_connections:
                p1_id = connection[0]
                p2_id = connection[1]

                p1_point = landmark_points.get(p1_id)
                p2_point = landmark_points.get(p2_id)
                #print(f"p1_id: {p1_id}, p2_id: {p2_id}, p1_point: {p1_point}, p2_point: {p2_point}")

                # Only add the line if both points are available (i.e., detected with valid depth)
                # and the distance is within a reasonable range
                distance_threshold = 0.15 # 15 cm (more than enough between possible landmarks)
                distance = math.sqrt(
                    (p1_point.x - p2_point.x) ** 2 +
                    (p1_point.y - p2_point.y) ** 2 +
                    (p1_point.z - p2_point.z) ** 2
                ) if p1_point and p2_point else float('inf')
                # print (f"distance: {distance}")

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
    node = HandVisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()