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

from msg_types.msg import HandLandmarks, Hand, Landmark

class HandTrackingNode(Node):
    def __init__(self):
        super().__init__('hand_tracking_node')

        # Declare parameters for output topics
        self.declare_parameter('output_landmarks_topic', '/hand_tracking/landmarks')
        self.declare_parameter('output_image_topic', '/hand_tracking/image_annotated')

        # Declare parameters for camera topics (what you want to subscribe to)
        self.color_topic_name = '/camera/camera/color/image_raw'
        # self.depth_topic_name = '/camera/camera/depth/image_rect_raw'
        self.depth_topic_name = '/camera/camera/aligned_depth_to_color/image_raw' # Aligned depth to color
        self.camera_info_topic_name = '/camera/camera/color/camera_info' # Use color info for depth aligned to color

        # Get the output topic names from parameters
        output_landmarks_topic = self.get_parameter('output_landmarks_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value

        # Initialize MediaPipe Hands and set parameters
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False, # Set to False for video stream, True for static images
            max_num_hands=2, # Maximum number of hands to detect and track simultaneously (1 or 2 usually)
            min_detection_confidence=0.5, # Min confidence for detection (1.0 = strict, 0.0 = lenient)
            min_tracking_confidence=0.5  # Min confidence for tracking (1.0 = strict, 0.0 = lenient)
        )
        self.mpDraw = mp.solutions.drawing_utils

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
            slop=0.1 # Max time difference (seconds) between messages to consider them synchronized
        )
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.landmark_pub = self.create_publisher(HandLandmarks, output_landmarks_topic, 10)
        self.image_pub = self.create_publisher(Image, output_image_topic, 10) # For annotated image

        self.get_logger().info("Hand Tracking Node Initialized")

    def camera_info_callback(self, msg):
        # Store camera intrinsics (using color topic)- Depth is 480x848 (hxw) and color is 480x640 (hxw)
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
            self.depth_scale = 0.0010000000474974513

    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.depth_scale is None:
            self.get_logger().warn("Waiting for Camera Info...")
            return

        try:
            # Convert ROS Images to OpenCV images
            color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # Depth image encoding from realsense2_camera is 16UC1 (unsigned 16-bit integer)
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough") # Passthrough for 16UC1

        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Process the color image with MediaPipe-- expects RGB, so convert from BGR
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(color_image_rgb)

        # Prepare the custom message
        hand_landmarks_msg = HandLandmarks()
        hand_landmarks_msg.header = color_msg.header # Use the timestamp and frame_id from the color frame

        # Creating copy of color image for drawing landmarks
        annotated_image = np.copy(color_image)

        if results.multi_hand_landmarks:
            # MediaPipe returns handedness (left/right) and landmark coordinates
            if len(results.multi_handedness) != len(results.multi_hand_landmarks):
                 self.get_logger().warn("Mismatch between hand landmarks and handedness results!")
                 # Attempt to process anyway, but this means result is unusual
            
            for i, handLms in enumerate(results.multi_hand_landmarks):
                # Get handedness
                hand_label = "Unknown"
                if i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i]
                    # Assuming the first classification is the primary one
                    if handedness.classification:
                        hand_label = handedness.classification[0].label
                    
                self.get_logger().debug(f"Processing {hand_label} Hand {i+1}")

                # Draw landmarks on the annotated image
                self.mpDraw.draw_landmarks(annotated_image, handLms, self.mpHands.HAND_CONNECTIONS)
                
                current_hand_msg = Hand()
                current_hand_msg.label = hand_label
                current_hand_msg.hand_id = i
                current_hand_msg.landmarks = [] 

                # Iterate through each landmark for the current hand
                for landmark_id, mp_landmark in enumerate(handLms.landmark):
                    # mp_landmark contains normalized coordinates (x, y, z - depth relative to wrist)
                    # Convert normalized (0 to 1) pixel coordinates to actual pixel coordinates
                    # Note: mp_landmark.z is relative depth, we use the Realsense depth image for absolute depth
                    px = int(mp_landmark.x * self.intrinsics['width'])
                    py = int(mp_landmark.y * self.intrinsics['height'])

                    # Ensure pixel coordinates are within bounds of the depth image
                    if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                        # Get the depth value at the landmark's pixel location
                        # The depth image might have a different resolution than the color image
                        # You would need to scale px and py if resolutions differ.
                        # Assuming same resolution for now (how this was originally done)
                        depth_value = depth_image[py, px]

                        # Realsense depth is typically in millimeters or has a specific scale factor
                        # The value is often 0 if depth is invalid (outside range, occluded, etc.)
                        if depth_value > 0: # Check for valid depth
                            # Convert depth value to meters using the depth scale
                            depth_in_meters = depth_value * self.depth_scale

                            # Convert 2D point (px, py) with depth to 3D point (x, y, z)
                            # Using camera intrinsic parameters
                            # Z coordinate is the depth itself
                            z = depth_in_meters
                            # X and Y coordinates derived using pinhole camera model
                            # x = (u - c_x) * z / f_x
                            # y = (v - c_y) * z / f_y
                            x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                            y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                            # Create the Landmark message
                            landmark_msg = Landmark()
                            landmark_msg.landmark_id = landmark_id
                            landmark_msg.point = Point()
                            landmark_msg.point.x = x
                            landmark_msg.point.y = y
                            landmark_msg.point.z = z # z is the distance from the camera plane

                            current_hand_msg.landmarks.append(landmark_msg)
                            self.get_logger().debug(f"  Landmark {landmark_id}: ({x:.3f}, {y:.3f}, {z:.3f}) m")
                        else:
                             self.get_logger().debug(f"  Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")
                             # Optionally add landmarks with NaN or a flag if depth is invalid

                    else:
                         self.get_logger().debug(f"  Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

                # Only add the hand if it has at least one valid 3D landmark
                if current_hand_msg.landmarks:
                    hand_landmarks_msg.hands.append(current_hand_msg)
                else:
                    self.get_logger().warn(f"Hand {i+1} ({hand_label}) detected by MediaPipe, but no valid 3D landmarks found.")


            # Publish the HandLandmarks message
            if hand_landmarks_msg.hands: # Only publish if any hands with valid landmarks were processed
                 self.landmark_pub.publish(hand_landmarks_msg)
                 self.get_logger().debug(f"Published {len(hand_landmarks_msg.hands)} hands.")
            else:
                 self.get_logger().debug("No hands with valid landmarks detected, not publishing.")


        # Publishing the annotated image
        try:
            # Convert annotated image back to ROS Image message
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_image_msg.header = color_msg.header # Use the same header
            self.image_pub.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Error publishing annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down hand tracking node...")
        node.hands.close() # Clean up MediaPipe resources
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()