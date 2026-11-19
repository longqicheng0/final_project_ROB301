
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

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

        self.create_subscription(
            Float64MultiArray,
            '/mean_img_rgb',
            self.rgb_callback,
            10
        )

        self.surface = None
        self.candidate = None
        self.stable_count = 0
        self.office_count = 0

        self.get_logger().info('Final Project initialized.')

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

        detected = self.classify_surface(*msg.data[:3])

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

        previous = self.surface
        self.surface = detected

        self.get_logger().info(f"Surface: {detected}")

        # Count only PATH -> OFFICE transitions
        if previous == "PATH" and detected == "OFFICE":
            self.office_count += 1
            self.get_logger().info(
                f"Office entered: {self.office_count}"
            )


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
