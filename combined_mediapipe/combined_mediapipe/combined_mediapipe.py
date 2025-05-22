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
from concurrent.futures import ThreadPoolExecutor, as_completed # Import for threading

# Import your custom message types
from msg_types.msg import HandLandmarks, Hand, FaceLandmarks, Face, Landmark, PoseLandmarks, Pose, PoseLandmark

class CombinedTrackingNode(Node):
    def __init__(self):
        super().__init__('combined_tracking_node')

        # Declare parameters for enabling/disabling MediaPipe modules
        self.declare_parameter('enable_hands', True)
        self.declare_parameter('enable_pose', True)
        self.declare_parameter('enable_face_mesh', True)

        # Get module enable flags from parameters
        self.enable_hands = self.get_parameter('enable_hands').get_parameter_value().bool_value
        self.enable_pose = self.get_parameter('enable_pose').get_parameter_value().bool_value
        self.enable_face_mesh = self.get_parameter('enable_face_mesh').get_parameter_value().bool_value

        self.get_logger().info(f"MediaPipe Modules Configuration:")
        self.get_logger().info(f"  Hands Enabled: {self.enable_hands}")
        self.get_logger().info(f"  Pose Enabled: {self.enable_pose}")
        self.get_logger().info(f"  Face Mesh Enabled: {self.enable_face_mesh}")

        # Initialize Thread Pool Executor for parallel MediaPipe processing
        # This executor is for your *MediaPipe tasks only*, not for ROS2 callbacks.
        # Max workers should ideally be number of CPU cores or slightly more for I/O bound tasks
        # For MediaPipe inference, it's often CPU-bound unless GPU is used, so cores is a good start.
        self.media_pipe_executor = ThreadPoolExecutor(max_workers=3) # Max 3 workers for Hand, Pose, Face

        # Declare parameters for output topics
        self.declare_parameter('output_hand_landmarks_topic', '/hand_landmarks')
        self.declare_parameter('output_pose_landmarks_topic', '/pose_landmarks')
        self.declare_parameter('output_face_landmarks_topic', '/face_landmarks')
        self.declare_parameter('output_image_topic', '/image_annotated')

        # Declare parameters for camera topics (what you want to subscribe to)
        self.color_topic_name = '/camera/camera/color/image_raw'
        self.depth_topic_name = '/camera/camera/aligned_depth_to_color/image_raw' # Aligned depth to color
        self.camera_info_topic_name = '/camera/camera/color/camera_info' # Use color info for depth aligned to color

        # Get the output topic names from parameters
        output_hand_landmarks_topic = self.get_parameter('output_hand_landmarks_topic').get_parameter_value().string_value
        output_pose_landmarks_topic = self.get_parameter('output_pose_landmarks_topic').get_parameter_value().string_value
        output_face_landmarks_topic = self.get_parameter('output_face_landmarks_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value

        # Initialize MediaPipe models conditionally
        self.mpHands = mp.solutions.hands
        self.hands = None
        if self.enable_hands:
            self.hands = self.mpHands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        self.mpPose = mp.solutions.pose
        self.pose = None
        if self.enable_pose:
            self.pose = self.mpPose.Pose(
                static_image_mode=False,
                model_complexity=1, # Adjust model complexity (0, 1, or 2)
                min_detection_confidence=0.8,
                min_tracking_confidence=0.8
            )
            self.mpPoseConnections = mp.solutions.pose.POSE_CONNECTIONS # Pose connections

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = None
        if self.enable_face_mesh:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        # Initialize MediaPipe Drawing Utilities (shared)
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

        # Time synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.image_callback)

        # Publishers
        self.hand_landmark_pub = self.create_publisher(HandLandmarks, output_hand_landmarks_topic, 10) if self.enable_hands else None
        self.pose_landmark_pub = self.create_publisher(PoseLandmarks, output_pose_landmarks_topic, 10) if self.enable_pose else None
        self.face_landmark_pub = self.create_publisher(FaceLandmarks, output_face_landmarks_topic, 10) if self.enable_face_mesh else None
        self.image_pub = self.create_publisher(Image, output_image_topic, 10) # Always publish annotated image

        self.get_logger().info("Combined Tracking Node Initialized")

    def camera_info_callback(self, msg):
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
            self.depth_scale = 0.0010000000474974513 # Realsense specific

    # --- New: Separate functions for MediaPipe processing ---
    def _process_hands(self, image_rgb):
        if self.hands:
            return self.hands.process(image_rgb)
        return None

    def _process_pose(self, image_rgb):
        if self.pose:
            return self.pose.process(image_rgb)
        return None

    def _process_face_mesh(self, image_rgb):
        if self.face_mesh:
            return self.face_mesh.process(image_rgb)
        return None
    # --- End New Functions ---


    def image_callback(self, color_msg, depth_msg):
        if self.intrinsics is None or self.depth_scale is None:
            self.get_logger().warn("Waiting for Camera Info...")
            return

        try:
            color_image = self.cv_bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, "passthrough")

        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Convert to RGB for MediaPipe and set as read-only
        # It's critical to make a COPY here for the image passed to other threads,
        # especially if MediaPipe's process() might modify the image or if other
        # threads try to read/write concurrently. While MediaPipe's process()
        # is generally safe with read-only flags, defensive copying is good.
        image_rgb_copy = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        image_rgb_copy.flags.writeable = False # Make it read-only for MediaPipe processing

        # --- Submit MediaPipe tasks to the thread pool ---
        futures = {}
        if self.enable_hands:
            futures['hands'] = self.media_pipe_executor.submit(self._process_hands, image_rgb_copy)
        if self.enable_pose:
            futures['pose'] = self.media_pipe_executor.submit(self._process_pose, image_rgb_copy)
        if self.enable_face_mesh:
            futures['face_mesh'] = self.media_pipe_executor.submit(self._process_face_mesh, image_rgb_copy)

        # Retrieve results from futures as they complete (or in a specific order)
        # We need all results before drawing, so we'll iterate through the known keys.
        hand_results = None
        pose_results = None
        face_results = None

        # Wait for all submitted tasks to complete
        for key, future in futures.items():
            try:
                if key == 'hands':
                    hand_results = future.result() # This will block until the 'hands' task is done
                elif key == 'pose':
                    pose_results = future.result() # This will block until the 'pose' task is done
                elif key == 'face_mesh':
                    face_results = future.result() # This will block until the 'face_mesh' task is done
            except Exception as exc:
                self.get_logger().error(f'{key} generated an exception: {exc}')
        
        # Prepare image for drawing (make writable and convert back to BGR)
        # Create a copy of the original color_image to draw on, ensuring it's writable
        annotated_image = np.copy(color_image) # Use original BGR image for drawing

        # --- Draw and Extract Hand Landmarks (conditionally) ---
        if self.enable_hands and hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks_msg = HandLandmarks()
            hand_landmarks_msg.header = color_msg.header

            if len(hand_results.multi_handedness) != len(hand_results.multi_hand_landmarks):
                 self.get_logger().warn("Mismatch between hand landmarks and handedness results!")

            for i, handLms in enumerate(hand_results.multi_hand_landmarks):
                hand_label = "Unknown"
                if i < len(hand_results.multi_handedness):
                    handedness = hand_results.multi_handedness[i]
                    if handedness.classification:
                        hand_label = handedness.classification[0].label
                    
                self.get_logger().debug(f"Processing {hand_label} Hand {i+1}")

                self.mp_drawing.draw_landmarks(annotated_image, handLms, self.mpHands.HAND_CONNECTIONS)
                
                current_hand_msg = Hand()
                current_hand_msg.label = hand_label
                current_hand_msg.hand_id = i
                current_hand_msg.landmarks = [] 

                for landmark_id, mp_landmark in enumerate(handLms.landmark):
                    px = int(mp_landmark.x * self.intrinsics['width'])
                    py = int(mp_landmark.y * self.intrinsics['height'])

                    if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                        depth_value = depth_image[py, px]
                        if depth_value > 0:
                            depth_in_meters = depth_value * self.depth_scale
                            z = depth_in_meters
                            x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                            y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                            landmark_msg = Landmark()
                            landmark_msg.landmark_id = landmark_id
                            landmark_msg.point = Point(x=x, y=y, z=z)
                            current_hand_msg.landmarks.append(landmark_msg)
                        else:
                             self.get_logger().debug(f"  Hand Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")
                    else:
                         self.get_logger().debug(f"  Hand Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

                if current_hand_msg.landmarks:
                    hand_landmarks_msg.hands.append(current_hand_msg)

            if hand_landmarks_msg.hands and self.hand_landmark_pub:
                 self.hand_landmark_pub.publish(hand_landmarks_msg)
                 self.get_logger().debug(f"Published {len(hand_landmarks_msg.hands)} hands.")
            elif self.enable_hands: # Only log if it's enabled but no hands found
                 self.get_logger().debug("No hands with valid landmarks detected, not publishing.")

        # --- Draw and Extract Pose Landmarks (conditionally) ---
        if self.enable_pose and pose_results and pose_results.pose_landmarks: 
            pose_landmarks_msg = PoseLandmarks()
            pose_landmarks_msg.header = color_msg.header

            self.mp_drawing.draw_landmarks(
                annotated_image,
                pose_results.pose_landmarks,
                self.mpPoseConnections, 
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            current_pose_msg = Pose()
            current_pose_msg.pose_id = 0 
            current_pose_msg.landmarks = []

            for landmark_id, mp_landmark in enumerate(pose_results.pose_landmarks.landmark):
                px = int(mp_landmark.x * self.intrinsics['width'])
                py = int(mp_landmark.y * self.intrinsics['height'])

                if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                    depth_value = depth_image[py, px]

                    if depth_value > 0:
                        depth_in_meters = depth_value * self.depth_scale
                        z = depth_in_meters
                        x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                        y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                        landmark_msg = PoseLandmark()
                        landmark_msg.landmark_id = landmark_id
                        landmark_msg.point = Point(x=x, y=y, z=z)
                        landmark_msg.visibility = mp_landmark.visibility # Include visibility
                        current_pose_msg.landmarks.append(landmark_msg)
                        self.get_logger().debug(f"  Landmark {landmark_id}: ({x:.3f}, {y:.3f}, {z:.3f}) m, Visibility: {mp_landmark.visibility:.2f}")
                    else:
                         self.get_logger().debug(f"  Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")

                else:
                     self.get_logger().debug(f"  Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

            if current_pose_msg.landmarks:
                pose_landmarks_msg.poses.append(current_pose_msg)
            elif self.enable_pose:
                 self.get_logger().warn("Pose detected by MediaPipe, but no valid 3D landmarks found.")

            if pose_landmarks_msg.poses and self.pose_landmark_pub:
                 self.pose_landmark_pub.publish(pose_landmarks_msg)
                 self.get_logger().debug(f"Published {len(pose_landmarks_msg.poses)} poses.")
            elif self.enable_pose:
                 self.get_logger().debug("No poses with valid landmarks detected, not publishing.")

        # --- Draw and Extract Face Landmarks (conditionally) ---
        if self.enable_face_mesh and face_results and face_results.multi_face_landmarks:
            face_landmarks_msg = FaceLandmarks()
            face_landmarks_msg.header = color_msg.header

            for i, faceLms in enumerate(face_results.multi_face_landmarks):
                self.get_logger().debug(f"Processing Face {i+1}")

                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=faceLms,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=faceLms,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())

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

                for landmark_id, mp_landmark in enumerate(faceLms.landmark):
                    px = int(mp_landmark.x * self.intrinsics['width'])
                    py = int(mp_landmark.y * self.intrinsics['height'])

                    if 0 <= px < self.intrinsics['width'] and 0 <= py < self.intrinsics['height']:
                        depth_value = depth_image[py, px]
                        if depth_value > 0:
                            depth_in_meters = depth_value * self.depth_scale
                            z = depth_in_meters
                            x = (px - self.intrinsics['ppx']) * z / self.intrinsics['fx']
                            y = (py - self.intrinsics['ppy']) * z / self.intrinsics['fy']

                            landmark_msg = Landmark()
                            landmark_msg.landmark_id = landmark_id
                            landmark_msg.point = Point(x=x, y=y, z=z)
                            current_face_msg.landmarks.append(landmark_msg)
                        else:
                             self.get_logger().debug(f"  Face Landmark {landmark_id} at ({px},{py}) has invalid depth ({depth_value}). Skipping.")
                    else:
                         self.get_logger().debug(f"  Face Landmark {landmark_id} pixel coords ({px},{py}) out of depth image bounds.")

                if current_face_msg.landmarks:
                    face_landmarks_msg.faces.append(current_face_msg)

            if face_landmarks_msg.faces and self.face_landmark_pub:
                 self.face_landmark_pub.publish(face_landmarks_msg)
                 self.get_logger().debug(f"Published {len(face_landmarks_msg.faces)} faces.")
            elif self.enable_face_mesh:
                 self.get_logger().debug("No faces with valid landmarks detected, not publishing.")
        
        # Publish the single annotated image
        try:
            annotated_image_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            annotated_image_msg.header = color_msg.header
            self.image_pub.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"CvBridge Error publishing annotated image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CombinedTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down combined tracking node...")
        # Shutdown the thread pool executor (your custom one for MediaPipe)
        node.media_pipe_executor.shutdown(wait=True) # Wait for active tasks to complete
        if node.hands:
            node.hands.close()
        if node.pose:
            node.pose.close()
        if node.face_mesh:
            node.face_mesh.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()