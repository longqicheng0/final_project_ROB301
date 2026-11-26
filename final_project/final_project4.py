
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

# --- 1.5 Delivery Mission ---

# Three office numbers assigned by the TA.
DELIVERY_TARGETS = [3, 7, 11]

# Minimum localization confidence required for delivery.
LOCALIZATION_THRESHOLD = 0.85


# --- 1.6 Delivery Maneuver ---

# Time to allow the robot to come to rest.
STOP_DURATION = 0.5

# Time spent facing the office to simulate mail delivery.
DELIVERY_PAUSE = 2.0

# Angular velocity magnitude (rad/s).
TURN_SPEED = 0.5

# Duration of 90-degree rotations (seconds).
TURN_LEFT_DURATION = np.pi / (2 * TURN_SPEED)
TURN_RIGHT_DURATION = np.pi / (2 * TURN_SPEED)

# --- 1.7 Mission State Definitions ---

class MissionState:

    NAVIGATING = "NAVIGATING"
    STOPPING = "STOPPING"
    TURNING_LEFT = "TURNING_LEFT"
    DELIVERING = "DELIVERING"
    TURNING_BACK = "TURNING_BACK"
    MISSION_COMPLETE = "MISSION_COMPLETE"

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
        # Mission execution state
        # ----------------------------------------------------

        self.mission_state = MissionState.NAVIGATING

        # Timestamp for timed state transitions.
        self.state_start_time = self.get_clock().now()

        # Delivery target management.
        self.remaining_targets = set(DELIVERY_TARGETS)
        self.delivered_targets = set()

        # Office currently receiving a delivery.
        self.active_target = None

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

        1. Ignore measurements during delivery.
        2. Classify surface and color.
        3. Confirm stable measurements.
        4. Detect valid PATH -> OFFICE transitions.
        5. Handle initial observation separately.
        """

        # Suspend perception and localization
        # during delivery or mission completion.
        if self.mission_state != MissionState.NAVIGATING:
            return

        if len(msg.data) < 3:
            return

        self.rgb_value = tuple(msg.data[:3])

        detected = self.classify_surface(
            *self.rgb_value
        )

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
    # 7.1 DELIVERY DECISION AND MISSION MANAGEMENT
    # ========================================================

    def check_delivery_target(self):
        """
        Determine whether the robot should initiate delivery.
        """

        # Only initiate delivery while navigating.
        if self.mission_state != MissionState.NAVIGATING:
            return

        # Robot must currently be on an office patch.
        if self.surface != "OFFICE":
            return

        # Localization must be available.
        if self.localization_confidence is None:
            return

        # Check whether the estimated office requires delivery.
        if self.estimated_office not in self.remaining_targets:
            return

        # Reject uncertain localization.
        if self.localization_confidence < LOCALIZATION_THRESHOLD:
            return

        # All conditions satisfied.
        self.start_delivery(self.estimated_office)


    def start_delivery(self, office):
        """
        Initialize the delivery maneuver at the target office.
        """

        self.active_target = office

        self.set_mission_state(MissionState.STOPPING)

        self.stop_robot()

        self.get_logger().info(
            f'Starting delivery at office {office}, '
            f'confidence={self.localization_confidence:.3f}'
        )


    def set_mission_state(self, new_state):
        """
        Change mission state and reset the state timer.
        """

        previous_state = self.mission_state

        self.mission_state = new_state
        self.state_start_time = self.get_clock().now()

        self.get_logger().info(
            f'Mission state: {previous_state} -> {new_state}'
        )


    # ========================================================
    # 7.2 TIMED DELIVERY EXECUTION
    # ========================================================

    def execute_delivery(self, command):
        """
        Execute the timed delivery maneuver.

        STOPPING -> TURNING_LEFT -> DELIVERING
        -> TURNING_BACK -> Finish delivery
        """

        elapsed = (
            self.get_clock().now() - self.state_start_time
        ).nanoseconds / 1e9

        # Default: stationary robot.
        command.twist.linear.x = 0.0
        command.twist.angular.z = 0.0

        # ----------------------------------------
        # 1. Stop before rotating
        # ----------------------------------------

        if self.mission_state == MissionState.STOPPING:

            if elapsed >= STOP_DURATION:
                self.set_mission_state(
                    MissionState.TURNING_LEFT
                )

        # ----------------------------------------
        # 2. Complete left rotation
        # ----------------------------------------

        elif self.mission_state == MissionState.TURNING_LEFT:

            if elapsed >= TURN_LEFT_DURATION:
                self.set_mission_state(
                    MissionState.DELIVERING
                )

        # ----------------------------------------
        # 3. Pause to simulate delivery
        # ----------------------------------------

        elif self.mission_state == MissionState.DELIVERING:

            if elapsed >= DELIVERY_PAUSE:
                self.set_mission_state(
                    MissionState.TURNING_BACK
                )

        # ----------------------------------------
        # 4. Complete right rotation
        # ----------------------------------------

        elif self.mission_state == MissionState.TURNING_BACK:

            if elapsed >= TURN_RIGHT_DURATION:
                self.finish_delivery()
                return

        # ----------------------------------------
        # 5. Generate motion command
        # ----------------------------------------

        if self.mission_state == MissionState.TURNING_LEFT:
            command.twist.angular.z = TURN_SPEED

        elif self.mission_state == MissionState.TURNING_BACK:
            command.twist.angular.z = -TURN_SPEED


    # ========================================================
    # 7.3 DELIVERY COMPLETION
    # ========================================================

    def finish_delivery(self):
        """
        Complete the current delivery and determine
        whether to resume navigation or stop.
        """

        # ----------------------------------------
        # 1. Update delivery records
        # ----------------------------------------

        office = self.active_target

        self.remaining_targets.remove(office)
        self.delivered_targets.add(office)

        self.active_target = None

        self.get_logger().info(
            f'Delivery completed at office {office}'
        )

        self.get_logger().info(
            f'Delivered: {sorted(self.delivered_targets)}, '
            f'Remaining: {sorted(self.remaining_targets)}'
        )

        # ----------------------------------------
        # 2. Check mission completion
        # ----------------------------------------

        if not self.remaining_targets:

            self.set_mission_state(
                MissionState.MISSION_COMPLETE
            )

            self.get_logger().info(
                'MISSION COMPLETE: All mail delivered.'
            )

            return

        # ----------------------------------------
        # 3. Reset sensor detection
        # ----------------------------------------

        self.surface = None
        self.rgb_value = None
        self.predicted_color = None
        self.line_index = None

        self.candidate_detection = None
        self.confirmed_detection = None
        self.stable_count = 0

        # Require PATH before accepting another office.
        self.ready_for_next_office = False

        # ----------------------------------------
        # 4. Resume navigation
        # ----------------------------------------

        self.set_mission_state(
            MissionState.NAVIGATING
        )


    # ========================================================
    # 8. MOTION CONTROL
    # ========================================================

    
    def control_loop(self):
        """
        Main motion controller.

        NAVIGATING       -> Line-following controller
        Delivery states  -> Timed delivery controller
        MISSION_COMPLETE -> Stop
        """

        command = TwistStamped()

        command.header.stamp = (
            self.get_clock().now().to_msg()
        )

        # ----------------------------------------
        # 1. Navigation
        # ----------------------------------------

        if self.mission_state == MissionState.NAVIGATING:

            self.navigate(command)

        # ----------------------------------------
        # 2. Mission completed
        # ----------------------------------------

        elif self.mission_state == MissionState.MISSION_COMPLETE:

            command.twist.linear.x = 0.0
            command.twist.angular.z = 0.0

        # ----------------------------------------
        # 3. Delivery maneuver
        # ----------------------------------------

        else:

            self.execute_delivery(command)

        # Publish the final velocity command.
        self.cmd_pub.publish(command)

    
    def navigate(self, command):
        """
        Execute normal navigation.

        OFFICE  -> Drive forward to find white line
        PATH    -> Proportional line following
        UNKNOWN -> Stop
        """

        # Default: stationary robot.
        command.twist.linear.x = 0.0
        command.twist.angular.z = 0.0

        # ----------------------------------------
        # 1. Drive through office patch
        # ----------------------------------------

        if self.surface == "OFFICE":

            command.twist.linear.x = LINEAR_SPEED

        # ----------------------------------------
        # 2. Follow white tape
        # ----------------------------------------

        elif (
            self.surface == "PATH"
            and self.line_index is not None
        ):

            error = self.line_index - LINE_CENTER

            command.twist.linear.x = LINEAR_SPEED

            command.twist.angular.z = (
                -LINE_KP * error
            )


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
