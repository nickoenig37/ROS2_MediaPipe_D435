import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters
import mediapipe as mp
import cv2
import numpy as np

# Import the new custom messages
from msg_types.msg import PoseLandmarks, Pose, PoseLandmark

class PoseTrackingNode(Node): # Renamed the class
    def __init__(self):
        super().__init__('pose_tracking_node') # Renamed the node

        # Declare parameters for output topics
        self.declare_parameter('output_landmarks_topic', '/pose_tracking/landmarks') # Updated topic name
        self.declare_parameter('output_image_topic', '/pose_tracking/image_annotated') # Updated topic name

        # Declare parameters for camera topics (what you want to subscribe to)
        self.color_topic_name = '/camera/camera/color/image_raw'
        self.depth_topic_name = '/camera/camera/aligned_depth_to_color/image_raw' # Aligned depth to color
        self.camera_info_topic_name = '/camera/camera/color/camera_info' # Use color info for depth aligned to color

        # Get the output topic names from parameters
        output_landmarks_topic = self.get_parameter('output_landmarks_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value

        # Initialize MediaPipe Pose and set parameters
        self.mpPose = mp.solutions.pose 
        self.pose = self.mpPose.Pose(
            static_image_mode=False,
            model_complexity=1, # You can adjust model complexity (0, 1, or 2)
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPoseConnections = mp.solutions.pose.POSE_CONNECTIONS # Pose connections

        self.cv_bridge = CvBridge()

        # Init vars for camera intrinsic parameters
        self.intrinsics = None
        self.depth_scale = None

        # Creating Subscribers for camera data-- we need to synchronize color and depth frames
        self.color_sub = message_filters.Subscriber(self, Image, self.color_topic_name)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic_name)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic_name, self.camera_info_callback, 10)

        # Time synchronizer-- ApproximateTimeSynchronizer is good if timestamps might be slightly off
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.landmark_pub = self.create_publisher(PoseLandmarks, output_landmarks_topic, 10) # Changed message type
        self.image_pub = self.create_publisher(Image, output_image_topic, 10)

        self.get_logger().info("Pose Tracking Node Initialized")

    def camera_info_callback(self, msg):
        # Store camera intrinsics (using color topic)
        if self.intrinsics is None:
            self.intrinsics = {
                'width': msg.width,
                'height': msg.height,
                'ppx': msg.k[2],
                'ppy': msg.k[5],
                'fx': msg.k[0],
                'fy': msg.k[4]
            }
            self.get_logger().info(f"Received Camera Info: {self.intrinsics}")
            # You might need to verify this depth scale for your specific Realsense model
            # A common value for 16UC1 depth is 0.001 (meters per unit)
            self.depth_scale = 0.001

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.depth_scale is None:
            self.get_logger().warn("Waiting for Camera Info...")
            return

        try:
            # Convert ROS Images to OpenCV images
            color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough") # Passthrough for 16UC1

        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Process the color image with MediaPipe-- expects RGB, so convert from BGR
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(color_image_rgb) # Changed from self.hands.process

        # Prepare the custom message
        pose_landmarks_msg = PoseLandmarks() # Changed message type
        pose_landmarks_msg.header = color_msg.header

        # Creating copy of color image for drawing landmarks
        annotated_image = np.copy(color_image)

        # MediaPipe Pose can return multiple poses in results.pose_landmarks
        if results.pose_landmarks: # Check if any poses are detected
            # MediaPipe Pose returns a single pose_landmarks object containing all landmarks
            # This code assumes a single dominant person is detected.

            # Draw landmarks on the annotated image
            self.mpDraw.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mpPoseConnections, # Use pose connections
                self.mpDraw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mpDraw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            current_pose_msg = Pose() # Create a Pose message for the detected person
            current_pose_msg.pose_id = 0 # Assign an ID (assuming single person for now)
            current_pose_msg.landmarks = []

            # Iterate through each landmark for the detected pose
            for landmark_id, mp_landmark in enumerate(results.pose_landmarks.landmark):
                # mp_landmark contains normalized coordinates (x, y, z - depth relative to hip)
                # and visibility/presence scores
                # Convert normalized (0 to 1) pixel coordinates to actual pixel coordinates
                px = int(mp_landmark.x * self.intrinsics['width'])
                py = int(mp_landmark.y * self.intrinsics['height'])

                # Ensure pixel coordinates are within bounds of the depth image
                if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                    # Get the depth value at the landmark's pixel location
                    depth_value = depth_image[py, px]

                    # Realsense depth is typically in millimeters or has a specific scale factor
                    if depth_value > 0: # Check for valid depth
                        # Convert depth value to meters using the depth scale
                        depth_in_meters = depth_value * self.depth_scale

                        # Convert 2D point (px, py) with depth to 3D point (x, y, z)
                        # Using camera intrinsic parameters
                        # Z coordinate is the depth itself
                        z = depth_in_meters
                        # X and Y coordinates derived using pinhole camera model
                        x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                        y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                        # Create the PoseLandmark message
                        landmark_msg = PoseLandmark() # Changed message type
                        landmark_msg.landmark_id = landmark_id
                        landmark_msg.point = Point()
                        landmark_msg.point.x = x
                        landmark_msg.point.y = y
                        landmark_msg.point.z = z # z is the distance from the camera plane
                        landmark_msg.visibility = mp_landmark.visibility # Add visibility

                        current_pose_msg.landmarks.append(landmark_msg)
                        self.get_logger().debug(f"  Landmark {landmark_id}: ({x:.3f}, {y:.3f}, {z:.3f}) m, Visibility: {mp_landmark.visibility:.2f}")
                    else:
                         self.get_logger().debug(f"  Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")

                else:
                     self.get_logger().debug(f"  Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

            # Only add the pose if it has at least one valid 3D landmark
            if current_pose_msg.landmarks:
                pose_landmarks_msg.poses.append(current_pose_msg) # Append the pose message
            else:
                 self.get_logger().warn("Pose detected by MediaPipe, but no valid 3D landmarks found.")


            # Publish the PoseLandmarks message
            if pose_landmarks_msg.poses: # Only publish if any poses with valid landmarks were processed
                 self.landmark_pub.publish(pose_landmarks_msg)
                 self.get_logger().debug(f"Published {len(pose_landmarks_msg.poses)} poses.")
            else:
                 self.get_logger().debug("No poses with valid landmarks detected, not publishing.")


        # Publishing the annotated image
        try:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_image_msg.header = color_msg.header
            self.image_pub.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Error publishing annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = PoseTrackingNode() # Changed node class
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down pose tracking node...")
        node.pose.close() # Clean up MediaPipe resources
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()