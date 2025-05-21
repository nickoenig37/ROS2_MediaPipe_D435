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

# Importing custom message types
from msg_types.msg import FaceLandmarks, Face, Landmark

class FaceMeshTrackingNode(Node):
    def __init__(self):
        super().__init__('face_mesh_tracking_node')

        # Declare parameters for output topics
        self.declare_parameter('output_landmarks_topic', '/face_mesh/landmarks')
        self.declare_parameter('output_image_topic', '/face_mesh/image_annotated')

        # Declare parameters for camera topics (to subscribe to)
        self.color_topic_name = '/camera/camera/color/image_raw'
        self.depth_topic_name = '/camera/camera/aligned_depth_to_color/image_raw' # Aligned depth to color
        self.camera_info_topic_name = '/camera/camera/color/camera_info' # Use color info for depth aligned to color

        # Get the output topic names from parameters
        output_landmarks_topic = self.get_parameter('output_landmarks_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value

        # Initialize MediaPipe Face Mesh and set parameters
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,             # Detect up to 1 face (change as needed)
            refine_landmarks=True,       # Enable attention mesh for more precise landmarks around eyes/lips
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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
        self.landmark_pub = self.create_publisher(FaceLandmarks, output_landmarks_topic, 10)
        self.image_pub = self.create_publisher(Image, output_image_topic, 10) # For annotated image

        self.get_logger().info("Face Mesh Tracking Node Initialized")

    def camera_info_callback(self, msg):
        # Store camera intrinsics (using color topic)- Depth and Color resolutions might differ
        # Ensure you use the correct dimensions for your specific camera setup.
        # For Realsense D435, if aligned_depth_to_color, depth image will have color frame's resolution.
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
            # This depth scale is specific to Realsense cameras with 16UC1 encoding
            # Verify this value for your specific camera model.
            self.depth_scale = 0.0010000000474974513

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
        color_image_rgb.flags.writeable = False # Read-only for MediaPipe processing

        results = self.face_mesh.process(color_image_rgb)

        # Prepare the custom message
        face_landmarks_msg = FaceLandmarks()
        face_landmarks_msg.header = color_msg.header # Use the timestamp and frame_id from the color frame

        # Creating copy of color image for drawing landmarks
        color_image_rgb.flags.writeable = True # Make writable again for drawing
        annotated_image = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR) # Convert back to BGR for OpenCV drawing

        if results.multi_face_landmarks:
            for i, faceLms in enumerate(results.multi_face_landmarks):
                self.get_logger().debug(f"Processing Face {i+1}")

                # Draw the full face mesh tesselation
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=faceLms,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                # Optionally, draw contours (eyes, lips, etc.)
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=faceLms,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())

                # Optionally, draw the irises (only if refine_landmarks=True)
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=faceLms,
                    connections=self.mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_iris_connections_style())
                
                current_face_msg = Face()
                current_face_msg.face_id = i
                current_face_msg.landmarks = [] 

                # Iterate through each landmark for the current face
                for landmark_id, mp_landmark in enumerate(faceLms.landmark):
                    # mp_landmark contains normalized coordinates (x, y, z - relative depth)
                    # Convert normalized (0 to 1) pixel coordinates to actual pixel coordinates
                    px = int(mp_landmark.x * self.intrinsics['width'])
                    py = int(mp_landmark.y * self.intrinsics['height'])

                    # Ensure pixel coordinates are within bounds of the depth image
                    if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                        depth_value = depth_image[py, px]

                        if depth_value > 0: # Check for valid depth
                            depth_in_meters = depth_value * self.depth_scale

                            z = depth_in_meters
                            x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                            y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                            # Create the Landmark message
                            landmark_msg = Landmark()
                            landmark_msg.landmark_id = landmark_id
                            landmark_msg.point = Point()
                            landmark_msg.point.x = x
                            landmark_msg.point.y = y
                            landmark_msg.point.z = z # z is the distance from the camera plane

                            current_face_msg.landmarks.append(landmark_msg)
                            self.get_logger().debug(f"  Landmark {landmark_id}: ({x:.3f}, {y:.3f}, {z:.3f}) m")
                        else:
                             self.get_logger().debug(f"  Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")
                    else:
                         self.get_logger().debug(f"  Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

                # Only add the face if it has at least one valid 3D landmark
                if current_face_msg.landmarks:
                    face_landmarks_msg.faces.append(current_face_msg)
                else:
                    self.get_logger().warn(f"Face {i+1} detected by MediaPipe, but no valid 3D landmarks found.")

            # Publish the FaceLandmarks message
            if face_landmarks_msg.faces: # Only publish if any faces with valid landmarks were processed
                 self.landmark_pub.publish(face_landmarks_msg)
                 self.get_logger().debug(f"Published {len(face_landmarks_msg.faces)} faces.")
            else:
                 self.get_logger().debug("No faces with valid landmarks detected, not publishing.")

        # Publishing the annotated image
        try:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_image_msg.header = color_msg.header # Use the same header
            self.image_pub.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Error publishing annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FaceMeshTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down Face Mesh tracking node...")
        node.face_mesh.close() # Clean up MediaPipe resources
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()