# ROS2 Package For MediaPipe Tracking Packages with Intel RealSense D435

### Package Information
This package has been designed and tested using ROS2 Humble and Ubuntu 22.04. It is intended to be used with the Intel RealSense D435 camera and MediaPipe with the included Packages.
- Note, this was also ran on a computer with an Intel Processor, Issues may arise with an AMD Processor when using the realsense(further mentioned in the Issues section).


## Setting up the Workspace
- Follow these steps starting from within your ros2_ws 
```bash
cd ~/ros2_ws/src
```
- Clone the repository for the Realsense D435 in ROS2
```bash
git clone https://github.com/IntelRealSense/realsense-ros.git
```
- Now go back into the workspace and install any dependencies and build+source the project
```bash
rosdep install --from-paths src --ignore-src -r -y
```
```bash
colcon build --symlink-install
```
```bash
source install/setup.bash
```

## Using the Programs (Launch/Run each in a new terminal)
- First and foremost, make sure the camera is connected to the computer and powered on.
- Then, run the following command to start the camera node (MUST BE IN Depth Alignment to work with MediaPipe Programs)
```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```
- These are the following Commands for running MediaPipe Tools and their cooresponding visualization tools:
### Hand Tracking
Mediapipe Hand Tracking
```bash
ros2 run mediapipe_hand_publisher hand_publisher
```
Hand Tracking Visualization in RVIZ2
```bash
ros2 run hand_visualization hand_visualization
```
### Body Tracking
Mediapipe Body Tracking
```bash
ros2 run body_pose_detector body_pose_detector
```
Body Tracking Visualization in RVIZ2
```bash
ros2 run skeleton_visualization skeleton_visualization
```


## Potential Issues you may face
1. **AMD Processors**: The code was designed and tested on an Intel Processor, so there may be issues with AMD Processors. 
- This issue can be summed up as follows:
    - `Intel’s Realsense library has a built-in ‘align’ function that performs the necessary calculations to overlay these images despite their FOV differences. While experimenting with this approach, though, we discovered that the align function isn’t optimized for non-Intel CPUs, which posed a problem for the ARM processor in our NVIDIA Jetson AGX Xavier. While aligning the depth and color frames during testing, our processing framerate dropped from the expected 30 FPS to just 2 to 4 FPS. Human arms tend to move fairly quickly, making framerate a critical issue in most hand tracking applications.`
    - This was discussed in the inspiration for some of this code found here- reference this for more information on the topic: https://medium.com/@smart-design-techology/hand-detection-in-3d-space-888433a1c1f3






