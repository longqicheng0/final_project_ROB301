
#!/usr/bin/env python3

import numpy as np
import rclpy

from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Float64MultiArray, UInt32


# ============================================================
# 1. CONFIGURATION — CHANGE THESE BEFORE RUNNING
# ============================================================

# --- 1.1 RGB Calibration ---

RGB_REFERENCES = {
    "orange": (246, 119, 93),
    "green":  (162, 171, 159),
    "yellow": (192, 163, 131),
    "blue":   (179, 180, 178),
    "tape":   (150, 132, 142),
    "floor":  (160, 150, 140)
}

COLOR_THRESHOLD = 25
STABLE_READINGS = 4


# --- 1.2 Line Following ---

LINE_CENTER = 320
LINEAR_SPEED = 0.06
LINE_KP = 0.004


# --- 1.3 Control Timing ---

CONTROL_PERIOD = 0.05
STATUS_PERIOD = 1.0


# --- 1.4 ROS Topics ---

RGB_TOPIC = '/mean_img_rgb'
LINE_TOPIC = '/line_idx'
CMD_TOPIC = '/cmd_vel'

BELIEF_TOPIC = '/office_belief'
ESTIMATE_TOPIC = '/estimated_office'


# ============================================================
# 2. BAYESIAN LOCALIZATION CONFIGURATION
# ============================================================

OFFICES = np.arange(2, 13)

# Practice topology: offices 2 through 12.
# Replace with final map when announced.
TOPOLOGY_COLORS = [
    "yellow", "green", "blue",
    "orange", "orange", "green",
    "blue", "orange", "yellow",
    "green", "blue"
]

MOTION_MODEL = {
    -1: (0.95, 0.02, 0.03),
     0: (0.02, 0.96, 0.02),
     1: (0.03, 0.02, 0.95)
}

SENSOR_MODEL = {
    "blue":    np.array([0.60, 0.20, 0.05, 0.05]),
    "green":   np.array([0.20, 0.60, 0.05, 0.05]),
    "yellow":  np.array([0.05, 0.05, 0.65, 0.20]),
    "orange":  np.array([0.05, 0.05, 0.15, 0.60]),
    "nothing": np.array([0.10, 0.10, 0.10, 0.10])
}

COLOR_INDEX = {
    "blue": 0,
    "green": 1,
    "yellow": 2,
    "orange": 3
}


# ============================================================
# 3. BAYESIAN LOCALIZER
# ============================================================

class BayesianLocalizer:

    def __init__(self):

        # Uniform initial belief.
        n = len(OFFICES)
        self.belief = np.full(n, 1.0 / n)

        # History for debugging and plotting.
        self.history = [self.belief.copy()]

    def update(self, observed_color, moved_forward):
        """
        Perform one Bayesian localization update.

        moved_forward = False:
            First office observation.
            Measurement update only.

        moved_forward = True:
            Robot traveled to the next office.
            Motion prediction + measurement update.
        """

        # ----------------------------------------
        # 1. Motion prediction
        # ----------------------------------------

        if moved_forward:

            reverse, stay, forward = MOTION_MODEL[1]

            previous = self.belief

            prediction = (
                forward * np.roll(previous, 1)
                + stay * previous
                + reverse * np.roll(previous, -1)
            )

        else:
            prediction = self.belief.copy()

        # ----------------------------------------
        # 2. Measurement likelihood
        # ----------------------------------------

        likelihood = np.array([
            SENSOR_MODEL[observed_color][COLOR_INDEX[color]]
            for color in TOPOLOGY_COLORS
        ])

        # ----------------------------------------
        # 3. Posterior update
        # ----------------------------------------

        posterior = prediction * likelihood
        total = posterior.sum()

        if total <= 0:
            raise ValueError(
                "Invalid Bayesian normalization"
            )

        self.belief = posterior / total
        self.history.append(self.belief.copy())

        # ----------------------------------------
        # 4. Estimate current office
        # ----------------------------------------

        best_index = int(np.argmax(self.belief))

        estimated_office = int(OFFICES[best_index])
        confidence = float(self.belief[best_index])

        return estimated_office, confidence


# ============================================================
# 4. ROBOT NODE
# ============================================================

class FinalProject(Node):

    def __init__(self):
        super().__init__('Final_Project')

        # ----------------------------------------------------
        # 4.1 Robot State
        # ----------------------------------------------------

        # Latest sensor measurements.
        self.line_index = None
        self.rgb_value = None
        self.predicted_color = None

        # Current surface: OFFICE, PATH, UNKNOWN.
        self.surface = None

        # Surface classification debounce.
        self.candidate_detection = None
        self.confirmed_detection = None
        self.stable_count = 0

        # Office detection.
        self.ready_for_next_office = False

        # Initial Bayesian observation.
        self.initial_observation_done = False

        # Bayesian localization.
        self.localizer = BayesianLocalizer()

        self.estimated_office = None
        self.localization_confidence = None

        # Robot execution state.
        self.done = False

        # ----------------------------------------------------
        # 4.2 ROS Subscribers
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
        # 4.3 ROS Publishers
        # ----------------------------------------------------

        self.cmd_pub = self.create_publisher(
            TwistStamped,
            CMD_TOPIC,
            10
        )

        self.belief_pub = self.create_publisher(
            Float64MultiArray,
            BELIEF_TOPIC,
            10
        )

        self.estimate_pub = self.create_publisher(
            UInt32,
            ESTIMATE_TOPIC,
            10
        )

        # ----------------------------------------------------
        # 4.4 Timers
        # ----------------------------------------------------

        self.control_timer = self.create_timer(
            CONTROL_PERIOD,
            self.control_loop
        )

        self.status_timer = self.create_timer(
            STATUS_PERIOD,
            self.log_status
        )

        self.get_logger().info(
            'Final Project initialized.'
        )

    # ========================================================
    # 5. SENSOR PROCESSING
    # ========================================================

    def line_callback(self, msg):
        """Store the latest white-line position."""

        self.line_index = msg.data

    def rgb_callback(self, msg):
        """
        Process RGB measurements.

        1. Classify surface and color.
        2. Confirm stable measurements.
        3. Detect valid PATH -> OFFICE transitions.
        4. Handle initial observation separately.
        """

        if len(msg.data) < 3:
            return

        self.rgb_value = tuple(msg.data[:3])

        detected = self.classify_surface(
            *self.rgb_value
        )

        if self.done:
            return

        # ----------------------------------------
        # 1. Preserve observed office color
        # ----------------------------------------

        observed_color = (
            self.predicted_color.lower()
            if detected == "OFFICE"
            else None
        )

        detection = (detected, observed_color)

        # ----------------------------------------
        # 2. Debounce classification
        # ----------------------------------------

        if detection == self.candidate_detection:
            self.stable_count += 1
        else:
            self.candidate_detection = detection
            self.stable_count = 1

        if self.stable_count < STABLE_READINGS:
            return

        # Ignore repeated identical detections.
        if detection == self.confirmed_detection:
            return

        self.confirmed_detection = detection
        self.surface = detected

        self.get_logger().info(
            f'Surface: {detected}, '
            f'color: {observed_color}'
        )

        # ----------------------------------------
        # 3. PATH detection
        # ----------------------------------------

        if detected == "PATH":

            # A confirmed PATH allows the next office
            # to trigger a new localization event.
            self.ready_for_next_office = True

        # ----------------------------------------
        # 4. OFFICE detection
        # ----------------------------------------

        elif detected == "OFFICE":

            if not self.initial_observation_done:

                # First confirmed office observation.
                #
                # It may be the starting patch, or the
                # first patch encountered from the path.
                #
                # Measurement update only.
                self.handle_initial_observation(
                    observed_color
                )

                # The current patch has been processed.
                self.ready_for_next_office = False

            elif self.ready_for_next_office:

                # Valid transition to the next office.
                self.ready_for_next_office = False

                self.handle_office_entry(
                    observed_color
                )

        # UNKNOWN:
        # Preserve ready_for_next_office so that brief
        # unknown readings do not cancel a valid
        # office entry.

    # ========================================================
    # 6. SURFACE CLASSIFICATION
    # ========================================================

    def classify_surface(self, r, g, b):
        """
        Find the closest calibrated RGB reference.

        Returns:
            PATH
            OFFICE
            UNKNOWN
        """

        rgb = (r, g, b)

        # ----------------------------------------
        # 1. Nearest-neighbor RGB classification
        # ----------------------------------------

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

        # ----------------------------------------
        # 2. Calculate RGB distance
        # ----------------------------------------

        distance_sq = sum(
            (measured - reference) ** 2
            for measured, reference in zip(
                rgb,
                RGB_REFERENCES[closest]
            )
        )

        if distance_sq > COLOR_THRESHOLD ** 2:
            return "UNKNOWN"

        # ----------------------------------------
        # 3. Determine surface
        # ----------------------------------------

        if closest in ("tape", "floor"):
            return "PATH"

        return "OFFICE"

    # ========================================================
    # 7. BAYESIAN LOCALIZATION INTEGRATION
    # ========================================================

    def handle_initial_observation(self, observed_color):
        """
        Process the first confirmed office color.

        Only perform the measurement update.
        No motion prediction is required because
        the initial belief is uniform.
        """

        self.get_logger().info(
            f'Initial office observation: {observed_color}'
        )

        estimated_office, confidence = self.localizer.update(
            observed_color,
            moved_forward=False
        )

        self.initial_observation_done = True

        self.publish_localization(
            estimated_office,
            confidence
        )

    def handle_office_entry(self, observed_color):
        """
        Called after a confirmed transition to
        the next office.

        1. Perform motion prediction.
        2. Apply observed color measurement.
        3. Publish localization results.
        """

        estimated_office, confidence = self.localizer.update(
            observed_color,
            moved_forward=True
        )

        self.publish_localization(
            estimated_office,
            confidence
        )

    def publish_localization(self, estimated_office, confidence):
        """
        Publish estimated office and belief distribution.
        """

        self.estimated_office = estimated_office
        self.localization_confidence = confidence

        # ----------------------------------------
        # 1. Publish full probability distribution
        # ----------------------------------------

        belief_msg = Float64MultiArray()
        belief_msg.data = self.localizer.belief.tolist()

        self.belief_pub.publish(belief_msg)

        # ----------------------------------------
        # 2. Publish estimated office
        # ----------------------------------------

        estimate_msg = UInt32()
        estimate_msg.data = estimated_office

        self.estimate_pub.publish(estimate_msg)

        # ----------------------------------------
        # 3. Log localization results
        # ----------------------------------------

        self.get_logger().info(
            f'Estimated office: {estimated_office}, '
            f'confidence: {confidence:.3f}'
        )

        self.get_logger().info(
            'Belief (offices 2-12): '
            f'{np.round(self.localizer.belief, 3)}'
        )

    # ========================================================
    # 8. MOTION CONTROL
    # ========================================================

    def control_loop(self):
        """
        Runs periodically.

        OFFICE  -> drive forward to find white line
        PATH    -> proportional line following
        UNKNOWN -> stop
        DONE    -> stop
        """

        command = TwistStamped()

        command.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # Default command is zero velocity.

        if self.done:
            pass

        elif self.surface == "OFFICE":

            command.twist.linear.x = LINEAR_SPEED

        elif (
            self.surface == "PATH"
            and self.line_index is not None
        ):

            error = self.line_index - LINE_CENTER

            command.twist.linear.x = LINEAR_SPEED

            command.twist.angular.z = (
                -LINE_KP * error
            )

        self.cmd_pub.publish(command)

    def stop_robot(self):
        """Publish a zero-velocity command."""

        command = TwistStamped()

        command.header.stamp = (
            self.get_clock().now().to_msg()
        )

        self.cmd_pub.publish(command)
    
    # ========================================================
    # 9. DEBUG / STATUS LOGGING
    # ========================================================

    def log_status(self):
        """Print robot state once per STATUS_PERIOD."""

        self.get_logger().info(
            f'surface={self.surface or "UNKNOWN"} '
            f'rgb={self.rgb_value} '
            f'predicted_color={self.predicted_color} '
            f'line_idx={self.line_index} '
            f'ready_for_next_office={self.ready_for_next_office} '
            f'initial_observed={self.initial_observation_done} '
            f'office_belief={np.round(self.localizer.belief, 3)} '
            f'estimated_office={self.estimated_office} '
            f'confidence={self.localization_confidence} '
            f'done={self.done}'
        )



# ============================================================
# 10. MAIN
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
