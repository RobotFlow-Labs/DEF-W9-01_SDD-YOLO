# DEF-sdd-yolo ROS2 Topic Contract

## Subscriptions
- `/camera/rgb/image` (sensor_msgs/Image): RGB frame input.

## Publications
- `/defense/sdd_yolo/detections` (custom DetectionFrame): normalized boxes + score + class.

## Notes
- This repository ships ROS2 adapter stubs only.
- Full rclpy node launch wiring is completed in deployment workspace.
