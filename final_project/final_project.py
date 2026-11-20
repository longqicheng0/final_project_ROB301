
#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64MultiArray, UInt32


# ============================================================
# 1. CONFIGURATION — CHANGE THESE BEFORE RUNNING
# ============================================================

# --- 1.1 RGB Calibration ---
# Calibrate these values for the actual lighting conditions.
# Format: (R, G, B)
RGB_REFERENCES = {
    "orange": (246, 119, 93),
    "green":  (162, 171, 159),
    "yellow": (192, 163, 131),
    "blue":   (179, 180, 178),
    "tape":   (150, 132, 142),
    "floor":  (160, 150, 140)
}

# Maximum Euclidean RGB distance for valid classification.
COLOR_THRESHOLD = 25

# Number of consecutive readings required to confirm a surface.
STABLE_READINGS = 4


# --- 1.2 Line Following ---
LINE_CENTER = 320         # Target pixel index
LINEAR_SPEED = 0.10       # Forward velocity (m/s)
LINE_KP = 0.004           # Proportional steering gain


# --- 1.3 Control Timing ---
CONTROL_PERIOD = 0.05     # 20 Hz
STATUS_PERIOD = 1.0       # Log status every 1 second


# --- 1.4 Route Testing ---
# TEMPORARY: Stop after entering this many patches.
# This is for full-route testing, NOT the final 3 deliveries.
PATCH_COUNT_LIMIT = 11


# --- 1.5 ROS Topics ---
RGB_TOPIC = '/mean_img_rgb'
LINE_TOPIC = '/line_idx'
CMD_TOPIC = '/cmd_vel'


# ============================================================
# 2. FINAL DEMO CONFIGURATION — TODO FOR DELIVERABLE 4.4
# ============================================================

# TODO: Add final office-color topology (offices 2 to 12).
# TODO: Add the three delivery office IDs assigned by the TA.
#
# These inputs must eventually connect to Bayesian localization
# and delivery decisions.
#
# IMPORTANT:
# The code below does not yet use a topological map or
# perform deliveries. It only follows the route and counts
# patch entries.


# ============================================================
# 3. ROBOT NODE
# ============================================================

class FinalProject(Node):

    def __init__(self):
        super().__init__('Final_Project')

        # ----------------------------------------------------
        # 3.1 Robot State
        # ----------------------------------------------------

        # Latest sensor measurements
        self.line_index = None
        self.rgb_value = None
        self.predicted_color = None

        # Surface classification
        # Possible values: OFFICE, PATH, UNKNOWN
        self.surface = None
        self.candidate_surface = None
        self.stable_count = 0

        # Route progress
        self.office_count = 0
        self.done = False

        # ----------------------------------------------------
        # 3.2 ROS Subscribers
        # ----------------------------------------------------

        self.create_subscription(
            Float64MultiArray,
            RGB_TOPIC,
            self.rgb_callback,
            10
        )

        self.create_subscription(
            UInt32,
            LINE_TOPIC,
            self.line_callback,
            10
        )

        # ----------------------------------------------------
        # 3.3 ROS Publisher
        # ----------------------------------------------------

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            CMD_TOPIC,
            10
        )

        # ----------------------------------------------------
        # 3.4 Timers
        # ----------------------------------------------------

        self.control_timer = self.create_timer(
            CONTROL_PERIOD,
            self.control_loop
        )

        self.status_timer = self.create_timer(
            STATUS_PERIOD,
            self.log_status
        )

        self.get_logger().info('Final Project initialized.')


    # ========================================================
    # 4. SENSOR PROCESSING
    # ========================================================

    def line_callback(self, msg):
        """Store the latest detected white-line position."""

        self.line_index = msg.data


    def rgb_callback(self, msg):
        """
        Process camera RGB data.

        1. Classify the surface.
        2. Filter unstable detections.
        3. Detect transitions between PATH and OFFICE.
        """

        if len(msg.data) < 3:
            return

        self.rgb_value = tuple(msg.data[:3])

        detected = self.classify_surface(*self.rgb_value)

        if self.done:
            return

        # --- Debounce surface classification ---
        if detected == self.candidate_surface:
            self.stable_count += 1
        else:
            self.candidate_surface = detected
            self.stable_count = 1

        if self.stable_count < STABLE_READINGS:
            return

        # Ignore repeated detections of the same surface.
        if detected == self.surface:
            return

        previous_surface = self.surface
        self.surface = detected

        self.get_logger().info(f'Surface: {detected}')

        if detected == "UNKNOWN":
            return

        # New office entry occurs at a PATH -> OFFICE transition.
        if previous_surface == "PATH" and detected == "OFFICE":
            self.handle_office_entry()


    # ========================================================
    # 5. SURFACE CLASSIFICATION
    # ========================================================

    def classify_surface(self, r, g, b):
        """
        Find the closest calibrated RGB reference.

        Returns:
            PATH    -> white tape or floor
            OFFICE  -> colored office patch
            UNKNOWN -> no sufficiently close reference
        """

        rgb = (r, g, b)

        # Nearest-neighbor RGB classification.
        closest = min(
            RGB_REFERENCES,
            key=lambda color: sum(
                (measured - reference) ** 2
                for measured, reference in zip(
                    rgb,
                    RGB_REFERENCES[color]
                )
            )
        )

        self.predicted_color = closest.upper()

        # Compute squared RGB distance.
        distance_sq = sum(
            (measured - reference) ** 2
            for measured, reference in zip(
                rgb,
                RGB_REFERENCES[closest]
            )
        )

        # Reject measurements outside the threshold.
        if distance_sq > COLOR_THRESHOLD ** 2:
            return "UNKNOWN"

        if closest in ("tape", "floor"):
            return "PATH"

        return "OFFICE"


    # ========================================================
    # 6. ROUTE PROGRESS / OFFICE DETECTION
    # ========================================================

    def handle_office_entry(self):
        """
        Called once when the robot enters a new colored patch.

        CURRENT:
            Count patch entries and stop after one full route.

        TODO FOR 4.4:
            1. Feed the observed color to Bayesian localization.
            2. Estimate the current office number.
            3. Check whether it is a delivery target.
            4. Trigger the delivery maneuver if required.
        """

        self.office_count += 1

        self.get_logger().info(
            f'Office entered: {self.office_count}'
        )

        # Temporary full-route testing condition.
        if self.office_count >= PATCH_COUNT_LIMIT:
            self.done = True

            self.get_logger().info(
                'Full route count reached. Stopping.'
            )


    # ========================================================
    # 7. MOTION CONTROL
    # ========================================================

    def control_loop(self):
        """
        Runs periodically.

        OFFICE  -> drive forward to find the white line
        PATH    -> proportional line following
        UNKNOWN -> stop
        DONE    -> stop
        """

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()

        # Default Twist values are zero (stop).
        if self.done:
            pass

        elif self.surface == "OFFICE":
            command.twist.linear.x = LINEAR_SPEED

        elif self.surface == "PATH" and self.line_index is not None:
            error = self.line_index - LINE_CENTER

            command.twist.linear.x = LINEAR_SPEED
            command.twist.angular.z = -LINE_KP * error

        self.cmd_pub.publish(command)


    def stop_robot(self):
        """Publish a zero-velocity command."""

        command = TwistStamped()
        command.header.stamp = self.get_clock().now().to_msg()

        self.cmd_pub.publish(command)


    # ========================================================
    # 8. DEBUG / STATUS LOGGING
    # ========================================================

    def log_status(self):
        """Print robot state once per STATUS_PERIOD."""

        self.get_logger().info(
            f'surface={self.surface or "UNKNOWN"} '
            f'rgb={self.rgb_value} '
            f'predicted_color={self.predicted_color} '
            f'line_idx={self.line_index} '
            f'office_count={self.office_count} '
            f'done={self.done}'
        )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO
    )

    node = FinalProject()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            node.stop_robot()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
