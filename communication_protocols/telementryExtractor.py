"""
telemetry_extractor_node.py - Jetson Onboard Data Extractor
Publishes structured JSON telemetry payloads from raw ROS 2 sensor topics.
"""
import time
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class TelemetryExtractorNode(Node):
    def __init__(self, robot_id: int):
        super().__init__(f'telemetry_extractor_{robot_id}')
        self.robot_id = robot_id
        
        # Sensor state cache
        self.x = 0.0
        self.y = 0.0
        self.linear_v = 0.0
        self.angular_v = 0.0
        self.current_intent = "IDLE"
        self.min_obstacle_dist = 999.0
        self.is_path_clear = True
        
        # Subscriptions to MiR100 hardware topics
        self.create_subscription(Odometry, f'/amr_{robot_id}/odom', self._odom_cb, 10)
        self.create_subscription(LaserScan, f'/amr_{robot_id}/scan', self._scan_cb, 10)

    def _odom_cb(self, msg: Odometry):
        self.x = round(msg.pose.pose.position.x, 2)
        self.y = round(msg.pose.pose.position.y, 2)
        self.linear_v = round(msg.twist.twist.linear.x, 2)
        self.angular_v = round(msg.twist.twist.angular.z, 2)

    def _scan_cb(self, msg: LaserScan):
        valid_ranges = [r for r in msg.ranges if msg.range_min <= r <= msg.range_max]
        if valid_ranges:
            self.min_obstacle_dist = round(min(valid_ranges), 2)
            self.is_path_clear = self.min_obstacle_dist > 1.0

    def set_intent(self, intent_str: str):
        self.current_intent = intent_str

    def get_json_payload(self) -> dict:
        """Returns JSON payload formatted for backend algorithm consumption."""
        return {
            "timestamp": time.time(),
            "amr_id": self.robot_id,
            "position": {"x": self.x, "y": self.y},
            "velocity": {"linear": self.linear_v, "angular": self.angular_v},
            "intent": self.current_intent,
            "lidar_summary": {
                "min_obstacle_dist": self.min_obstacle_dist,
                "path_clear": self.is_path_clear
            }
        }