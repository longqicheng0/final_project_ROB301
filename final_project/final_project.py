#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, UInt32


class FinalProject(Node):
    def __init__(self):
        super().__init__('Final_Project')

        # Calibrated RGB values
        self.rgb_ref = {
            "orange": (246, 119, 93),
            "green":  (162, 171, 159),
            "yellow": (192, 163, 131),
            "blue":   (179, 180, 178),
            "tape":   (150, 132, 142),
            "floor":  (160, 150, 140)
        }

        # Proportional line-following settings
        self.line_center = 320
        self.linear_speed = 0.1
        self.k_p = 0.004
        self.line_index = None
        self.rgb_value = None

        self.create_subscription(
            Float64MultiArray,
            '/mean_img_rgb',
            self.rgb_callback,
            10
        )

        self.create_subscription(
            UInt32,
            '/line_idx',
            self.line_callback,
            10
        )
        self.cmd_pub = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )
        self.control_timer = self.create_timer(0.05, self.control_loop)
        self.status_timer = self.create_timer(1.0, self.log_status)

        self.surface = None
        self.candidate = None
        self.stable_count = 0
        self.office_count = 0
        self.office_target = 11
        self.was_patch = True
        self.done = False

        self.get_logger().info('Final Project initialized.')

    def line_callback(self, msg):
        self.line_index = msg.data

    def log_status(self):
        surface = {
            "OFFICE": "PATCH",
            "PATH": "LINE"
        }.get(self.surface, "UNKNOWN")

        self.get_logger().info(
            f"surface={surface} rgb={self.rgb_value} "
            f"line_idx={self.line_index} office_count={self.office_count}"
        )

    def control_loop(self):
        if self.done:
            stop_command = TwistStamped()
            stop_command.header.stamp = self.get_clock().now().to_msg()
            self.cmd_pub.publish(stop_command)
            return

        if self.line_index is None:
            return

        error = self.line_index - self.line_center

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.twist.linear.x = self.linear_speed
        command.twist.angular.z = -self.k_p * error

        self.cmd_pub.publish(command)

    def classify_surface(self, r, g, b):
        rgb = (r, g, b)

        # Find the closest RGB reference
        closest = min(
            self.rgb_ref,
            key=lambda c: sum(
                (a - b) ** 2
                for a, b in zip(rgb, self.rgb_ref[c])
            )
        )

        distance_sq = sum(
            (a - b) ** 2
            for a, b in zip(rgb, self.rgb_ref[closest])
        )

        if distance_sq > 25 ** 2:
            return "UNKNOWN"

        if closest in ("tape", "floor"):
            return "PATH"

        return "OFFICE"

    def rgb_callback(self, msg):
        if len(msg.data) < 3:
            return

        self.rgb_value = tuple(msg.data[:3])

        if self.done:
            return

        detected = self.classify_surface(*self.rgb_value)

        # Require 4 consistent readings
        if detected == self.candidate:
            self.stable_count += 1
        else:
            self.candidate = detected
            self.stable_count = 1

        if self.stable_count < 4:
            return

        if detected == "UNKNOWN" or detected == self.surface:
            return

        self.surface = detected

        self.get_logger().info(f"Surface: {detected}")

        is_patch = detected == "OFFICE"

        # Starting on an office does not count as a new entry.
        if is_patch and not self.was_patch:
            self.office_count += 1
            self.get_logger().info(
                f"Office entered: {self.office_count}"
            )

            if self.office_count >= self.office_target:
                self.done = True
                self.get_logger().info('11 offices reached. Stopping.')

        self.was_patch = is_patch


def main():
    rclpy.init()
    node = FinalProject()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
