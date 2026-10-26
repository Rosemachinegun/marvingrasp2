#!/usr/bin/env python3
"""Low-speed right-arm greeting gesture for the Apex robot.

The script is deliberately dry-run by default. Pass --execute to command the
real robot. It claims joint_cmd_mux input 4 (replay), streams timestamped
JointcmdArm commands, and restores the previous mux input when finished.
"""

import argparse
import math
import sys
import time
from collections.abc import Callable, Iterable
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import Int32

from grasp_core.communication.marvin_interfaces import load_marvin_interfaces


JointcmdArm, Jointfeedback, Int = load_marvin_interfaces()


# Current request-IK HOME joint positions, radians.  The old wave recording's
# P0 no longer matches the application's HOME target, so this was measured at
# the settled startup HOME on 2026-10-23. arm_positions is [left 7, right 7].
P0_HOME = [-2.9408797896, -1.2749169992, 1.6755090245, -2.3479754424,
           0.2397389098, -0.2999804958, -0.1416123093]
P1_READY = [-1.9793169822, -1.2610451257, 1.6515223321, -1.9429166301,
            0.1983861434, -0.9749507179, -0.3569497726]
# The two manually captured side poses were on the same side of P1. Build a
# wave around P1 instead, using the known-safe P1->old-P2 direction. The
# opposite/right side is intentionally smaller after the first robot test.
LEFT_WAVE_SCALE = 0.70
RIGHT_WAVE_SCALE = 0.45
P2_SIDE_A = [-1.8120288671, -1.1262941413, 1.4821815324, -2.0091112833,
             0.0065920078, -0.7538584864, 0.0090950796]
P3_SIDE_B = [-2.0868593419, -1.3476707585, 1.7603842748, -1.9003629245,
             0.3216823734, -1.1170814382, -0.5922643205]

LEFT_REPLAY_TOPIC = '/control/replay/joint_cmd_A'
RIGHT_REPLAY_TOPIC = '/control/replay/joint_cmd_B'
JOINT_FEEDBACK_TOPIC = '/info/joint_feedback'
INPUT_MODE_TOPIC = '/control/input_mode'
SET_INPUT_SERVICE = '/control/set_input'

REPLAY_INPUT = 4
REQUEST_IK_INPUT = 1
MAX_START_TO_READY_DELTA_RAD = 1.50
HOME_START_TOLERANCE_RAD = 0.08
STATIONARY_VELOCITY_RAD_S = 0.02
STATIONARY_DURATION_SEC = 0.20
REPLAY_PRIME_SEC = 0.08
FINAL_HOME_TOLERANCE_RAD = 0.03
FINAL_HOME_TIMEOUT_SEC = 4.0
DEFAULT_MAX_VELOCITY_RAD_S = 1.00
DEFAULT_MAX_ACCELERATION_RAD_S2 = 3.00
WAVE_MIN_DURATION_SEC = 0.18
HOME_TRANSITION_MIN_DURATION_SEC = 1.50
MOVE_ENDPOINT_HOLD_SEC = 0.01


class WaveStopped(RuntimeError):
    """Raised when the application requests a stop during the gesture."""


class WaveRecoveryRequired(WaveStopped):
    """The gesture stopped away from HOME and replay remains selected."""

    manual_recovery_required = True


def minimum_jerk(u: float) -> float:
    """Quintic blend with zero velocity and acceleration at both ends."""
    u = max(0.0, min(1.0, u))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def safe_duration(start: Iterable[float], target: Iterable[float],
                  max_velocity: float, max_acceleration: float,
                  minimum: float) -> float:
    """Duration that bounds a minimum-jerk profile's velocity/acceleration."""
    max_delta = max(abs(b - a) for a, b in zip(start, target))
    velocity_time = 1.875 * max_delta / max(max_velocity, 1e-6)
    acceleration_time = math.sqrt(5.774 * max_delta / max(max_acceleration, 1e-6))
    return max(minimum, velocity_time, acceleration_time)


class WaveGreeting(Node):
    def __init__(
        self,
        rate_hz: float,
        *,
        stop_requested: Callable[[], bool] | None = None,
    ):
        super().__init__('wave_greeting')
        self.rate_hz = rate_hz
        self.stop_requested = stop_requested or (lambda: False)
        self.feedback: Optional[Jointfeedback] = None
        self.feedback_sequence = 0
        self.input_mode: Optional[int] = None
        self.required_input_mode: int | None = None
        self.left_hold_pose: List[float] | None = None
        self.missing_interfaces: List[str] = []

        self.left_cmd_pub = self.create_publisher(
            JointcmdArm, LEFT_REPLAY_TOPIC, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(
            JointcmdArm, RIGHT_REPLAY_TOPIC, qos_profile_sensor_data)
        self.feedback_sub = self.create_subscription(
            Jointfeedback, JOINT_FEEDBACK_TOPIC, self._feedback_cb,
            qos_profile_sensor_data)

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.mode_sub = self.create_subscription(
            Int32, INPUT_MODE_TOPIC, self._mode_cb, latched_qos)
        self.mode_client = self.create_client(Int, SET_INPUT_SERVICE)

    def _feedback_cb(self, msg: Jointfeedback) -> None:
        self.feedback = msg
        self.feedback_sequence += 1

    def _mode_cb(self, msg: Int32) -> None:
        self.input_mode = int(msg.data)

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and rclpy.ok():
            self.raise_if_stopped()
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.feedback is not None and len(self.feedback.arm_positions) == 14
                    and len(self.feedback.arm_velocities) == 14
                    and self.input_mode is not None
                    and self.mode_client.service_is_ready()
                    and self.left_cmd_pub.get_subscription_count() > 0
                    and self.cmd_pub.get_subscription_count() > 0):
                return True
        missing = []
        if self.feedback is None:
            missing.append(JOINT_FEEDBACK_TOPIC)
        elif len(self.feedback.arm_positions) != 14:
            missing.append(
                f'14-element joint feedback (got {len(self.feedback.arm_positions)})')
        elif len(self.feedback.arm_velocities) != 14:
            missing.append(
                f'14-element joint velocity feedback '
                f'(got {len(self.feedback.arm_velocities)})')
        if self.input_mode is None:
            missing.append(INPUT_MODE_TOPIC)
        if not self.mode_client.service_is_ready():
            missing.append(SET_INPUT_SERVICE)
        if self.left_cmd_pub.get_subscription_count() <= 0:
            missing.append(f'{LEFT_REPLAY_TOPIC} subscriber')
        if self.cmd_pub.get_subscription_count() <= 0:
            missing.append(f'{RIGHT_REPLAY_TOPIC} subscriber')
        self.missing_interfaces = missing
        self.get_logger().error('Missing: ' + ', '.join(missing))
        return False

    def raise_if_stopped(self) -> None:
        if self.stop_requested():
            raise WaveStopped("wave stopped by application")
        if (self.required_input_mode is not None
                and self.input_mode != self.required_input_mode):
            raise WaveStopped(
                "wave stopped: mux input changed externally "
                f"to {self.input_mode}"
            )

    def set_input(
        self,
        mode: int,
        timeout: float = 3.0,
        *,
        keepalive_pose: Iterable[float] | None = None,
    ) -> bool:
        if not self.mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f'{SET_INPUT_SERVICE} is unavailable')
            return False
        request = Int.Request()
        request.data = mode
        future = self.mode_client.call_async(request)
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            self.raise_if_stopped()
            if keepalive_pose is not None:
                self.publish_pose(keepalive_pose)
            rclpy.spin_once(self, timeout_sec=0.05)
        if keepalive_pose is not None:
            self.publish_pose(keepalive_pose)
        if not future.done():
            self.get_logger().error(f'set_input({mode}) timed out')
            return False
        response = future.result()
        if response is None or not response.success:
            message = '' if response is None else response.message
            self.get_logger().error(f'set_input({mode}) failed: {message}')
            return False
        return True

    def wait_for_input_mode(
        self,
        mode: int,
        *,
        timeout: float = 1.0,
        keepalive_pose: Iterable[float] | None = None,
    ) -> bool:
        """Wait for the latched mux echo while holding a no-jump command."""
        deadline = time.monotonic() + timeout
        period = 1.0 / self.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            self.raise_if_stopped()
            if keepalive_pose is not None:
                self.publish_pose(keepalive_pose)
            rclpy.spin_once(self, timeout_sec=min(period, 0.02))
            if self.input_mode == mode:
                return True
        return False

    def wait_for_safe_home_start(
        self,
        timeout: float = 3.0,
    ) -> tuple[List[float], List[float]]:
        """Require both arms still and the right arm at current app HOME."""
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        last_sequence = -1
        while rclpy.ok() and time.monotonic() < deadline:
            self.raise_if_stopped()
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.feedback is None or self.feedback_sequence == last_sequence:
                continue
            last_sequence = self.feedback_sequence
            all_positions = list(self.feedback.arm_positions)
            all_velocities = list(self.feedback.arm_velocities)
            if not all(math.isfinite(value)
                       for value in all_positions + all_velocities):
                raise RuntimeError("non-finite arm joint feedback")
            left_positions = all_positions[:7]
            positions = all_positions[7:14]

            home_errors = [abs(actual - expected)
                           for actual, expected in zip(positions, P0_HOME)]
            max_home_error = max(home_errors)
            if max_home_error > HOME_START_TOLERANCE_RAD:
                joint = home_errors.index(max_home_error) + 1
                raise RuntimeError(
                    "wave refused: right arm is not at HOME "
                    f"(joint {joint} error={max_home_error:.3f}rad, "
                    f"limit={HOME_START_TOLERANCE_RAD:.3f}rad)"
                )

            ready_errors = [abs(actual - expected)
                            for actual, expected in zip(positions, P1_READY)]
            max_ready_delta = max(ready_errors)
            if max_ready_delta > MAX_START_TO_READY_DELTA_RAD:
                joint = ready_errors.index(max_ready_delta) + 1
                raise RuntimeError(
                    "wave refused: measured HOME is too far from wave READY "
                    f"(joint {joint} delta={max_ready_delta:.3f}rad, "
                    f"limit={MAX_START_TO_READY_DELTA_RAD:.3f}rad)"
                )

            # Mux selection is shared by both arms. The left arm will be held
            # on its measured pose, so it must also be stationary at takeover.
            max_speed = max(abs(value) for value in all_velocities)
            if max_speed <= STATIONARY_VELOCITY_RAD_S:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= STATIONARY_DURATION_SEC:
                    self.get_logger().info(
                        "Stationary measured HOME confirmed: "
                        f"home_error={max_home_error:.4f}rad "
                        f"start_to_ready={max_ready_delta:.4f}rad "
                        f"max_speed={max_speed:.4f}rad/s"
                    )
                    return left_positions, positions
            else:
                stable_since = None
        raise RuntimeError(
            "wave refused: right arm did not remain stationary at HOME"
        )

    def hold_until_measured_target(
        self,
        target: List[float],
        *,
        tolerance: float,
        timeout: float,
    ) -> None:
        """Keep commanding a target until feedback confirms it is stationary."""
        deadline = time.monotonic() + timeout
        stable_since: float | None = None
        last_sequence = -1
        period = 1.0 / self.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            self.raise_if_stopped()
            self.publish_pose(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            if (self.feedback is not None
                    and self.feedback_sequence != last_sequence):
                last_sequence = self.feedback_sequence
                all_positions = list(self.feedback.arm_positions)
                all_velocities = list(self.feedback.arm_velocities)
                right_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(all_positions[7:14], target)
                )
                left_error = (
                    max(
                        abs(actual - expected)
                        for actual, expected in zip(
                            all_positions[:7], self.left_hold_pose,
                        )
                    )
                    if self.left_hold_pose is not None
                    else 0.0
                )
                max_error = max(left_error, right_error)
                max_speed = max(abs(value) for value in all_velocities)
                if (max_error <= tolerance
                        and max_speed <= STATIONARY_VELOCITY_RAD_S):
                    stable_since = stable_since or time.monotonic()
                    if time.monotonic() - stable_since >= STATIONARY_DURATION_SEC:
                        return
                else:
                    stable_since = None
            time.sleep(period)
        raise RuntimeError(
            "right arm did not settle at its measured starting HOME pose"
        )

    def publish_pose(self, positions: Iterable[float]) -> None:
        stamp = self.get_clock().now().to_msg()
        if self.left_hold_pose is not None:
            left_msg = JointcmdArm()
            left_msg.header.stamp = stamp
            left_msg.positions = list(self.left_hold_pose)
            self.left_cmd_pub.publish(left_msg)
        right_msg = JointcmdArm()
        right_msg.header.stamp = stamp
        right_msg.positions = list(positions)
        self.cmd_pub.publish(right_msg)

    def hold(self, pose: List[float], duration: float) -> None:
        deadline = time.monotonic() + duration
        period = 1.0 / self.rate_hz
        while rclpy.ok() and time.monotonic() < deadline:
            self.raise_if_stopped()
            self.publish_pose(pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def move(self, start: List[float], target: List[float], duration: float) -> None:
        self.get_logger().info(f'Move duration: {duration:.2f} s')
        begin = time.monotonic()
        period = 1.0 / self.rate_hz
        while rclpy.ok():
            self.raise_if_stopped()
            elapsed = time.monotonic() - begin
            u = min(elapsed / duration, 1.0)
            blend = minimum_jerk(u)
            pose = [a + (b - a) * blend for a, b in zip(start, target)]
            self.publish_pose(pose)
            rclpy.spin_once(self, timeout_sec=0.0)
            if u >= 1.0:
                break
            time.sleep(period)
        self.hold(target, MOVE_ENDPOINT_HOLD_SEC)


def execute_wave(
    *,
    repetitions: int = 1,
    max_velocity: float = DEFAULT_MAX_VELOCITY_RAD_S,
    max_acceleration: float = DEFAULT_MAX_ACCELERATION_RAD_S2,
    rate_hz: float = 100.0,
    stop_requested: Callable[[], bool] | None = None,
) -> str:
    """Execute the captured right-arm wave inside an existing ROS process.

    The caller owns ``rclpy`` initialization/shutdown.  This makes the gesture
    usable by the main application's serialized robot-action worker without
    disturbing the request-IK node's ROS context.
    """
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if rate_hz <= 0.0:
        raise ValueError("rate_hz must be > 0")
    if not rclpy.ok():
        raise RuntimeError("ROS2 is not initialized")

    node = WaveGreeting(rate_hz, stop_requested=stop_requested)
    previous_mode: Optional[int] = None
    replay_claimed = False
    safe_to_restore = False
    try:
        if not node.wait_until_ready():
            missing = ", ".join(node.missing_interfaces) or "unknown interface"
            raise RuntimeError(
                f"required ROS interface unavailable: {missing}"
            )
        previous_mode = node.input_mode
        if previous_mode != REQUEST_IK_INPUT:
            raise RuntimeError(
                "wave refused: expected request-IK mux input "
                f"{REQUEST_IK_INPUT}, got {previous_mode}"
            )
        left_hold, current = node.wait_for_safe_home_start()
        node.left_hold_pose = left_hold
        node.get_logger().info(f'Current mux input: {previous_mode}')

        # Prime DDS discovery/queues with the exact measured pose. The mux
        # currently drops inactive-source samples, so keep publishing the same
        # pose while the service switches and until mode=4 is echoed.
        node.hold(current, REPLAY_PRIME_SEC)
        if not node.set_input(REPLAY_INPUT, keepalive_pose=current):
            raise RuntimeError(f"failed to select replay mux input {REPLAY_INPUT}")
        replay_claimed = previous_mode != REPLAY_INPUT
        if not node.wait_for_input_mode(REPLAY_INPUT, keepalive_pose=current):
            raise RuntimeError("replay mux input was not confirmed")
        node.required_input_mode = REPLAY_INPUT
        safe_to_restore = True
        node.hold(current, REPLAY_PRIME_SEC)

        def go(
            start: List[float],
            target: List[float],
            minimum: float = WAVE_MIN_DURATION_SEC,
        ) -> None:
            duration = safe_duration(
                start, target, max_velocity, max_acceleration, minimum
            )
            node.move(start, target, duration)

        def wave_batch() -> None:
            go(P1_READY, P2_SIDE_A)
            for repetition in range(repetitions):
                go(P2_SIDE_A, P3_SIDE_B)
                if repetition + 1 < repetitions:
                    go(P3_SIDE_B, P2_SIDE_A)

        safe_to_restore = False
        go(current, P1_READY, HOME_TRANSITION_MIN_DURATION_SEC)
        node.hold(P1_READY, 0)
        wave_batch()

        # Return to the exact measured starting configuration, not the nominal
        # capture, then verify the hardware is settled before handing mux back.
        go(P3_SIDE_B, current, HOME_TRANSITION_MIN_DURATION_SEC)
        node.hold_until_measured_target(
            current,
            tolerance=FINAL_HOME_TOLERANCE_RAD,
            timeout=FINAL_HOME_TIMEOUT_SEC,
        )
        safe_to_restore = True
        status = f"Right-arm wave completed (repetitions={repetitions})"
        node.get_logger().info(status)
        return status
    except (Exception, KeyboardInterrupt) as exc:
        if replay_claimed and not safe_to_restore and node.input_mode == REPLAY_INPUT:
            raise WaveRecoveryRequired(
                f"{exc}; mux remains on replay because the arm is away from HOME"
            ) from exc
        raise
    finally:
        # Hand mux back only while the arm is still/again at the measured HOME.
        # Away from HOME, an old IK target could cause an abrupt return.
        # Ignore the application stop callback only for this safe cleanup.
        node.stop_requested = lambda: False
        node.required_input_mode = None
        if (previous_mode is not None
                and (not replay_claimed or safe_to_restore)
                and node.mode_client.service_is_ready()):
            if node.set_input(previous_mode):
                node.get_logger().info(f'Restored mux input to {previous_mode}')
            else:
                node.get_logger().error(
                    f'Failed to restore mux input to {previous_mode}'
                )
        elif replay_claimed and node.input_mode == REPLAY_INPUT:
            node.get_logger().error(
                "Wave stopped away from HOME; leaving mux on replay to avoid "
                "an abrupt jump to a stale command. Recover the arm manually."
            )
        elif replay_claimed:
            node.get_logger().error(
                f"Wave mux was taken externally by input {node.input_mode}; "
                "not overriding that selection."
            )
        node.destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true',
                        help='actually command the real robot')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='number of P2/P3 wave repetitions')
    parser.add_argument(
        '--max-velocity', type=float, default=DEFAULT_MAX_VELOCITY_RAD_S,
        help='conservative joint velocity limit, rad/s',
    )
    parser.add_argument(
        '--max-acceleration', type=float,
        default=DEFAULT_MAX_ACCELERATION_RAD_S2,
        help='conservative joint acceleration limit, rad/s^2',
    )
    parser.add_argument('--rate', type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        print('repetitions must be >= 1', file=sys.stderr)
        return 2

    print('Planned gesture: measured HOME -> P1 -> '
          f'wave x{args.repetitions} -> measured HOME')
    print('Wave endpoints around P1: '
          f'left scale={LEFT_WAVE_SCALE:.2f}, right scale={RIGHT_WAVE_SCALE:.2f}')
    print(f'Limits: velocity={args.max_velocity} rad/s, '
          f'acceleration={args.max_acceleration} rad/s^2')
    if not args.execute:
        print('DRY RUN ONLY. Add --execute to move the real robot.')
        return 0

    rclpy.init()
    try:
        execute_wave(
            repetitions=args.repetitions,
            max_velocity=args.max_velocity,
            max_acceleration=args.max_acceleration,
            rate_hz=args.rate,
        )
        return 0
    except (KeyboardInterrupt, WaveStopped):
        print(
            'Interrupted: stopped streaming immediately; robot was not auto-moved home',
            file=sys.stderr,
        )
        return 130
    except RuntimeError as exc:
        print(f'Wave failed: {exc}', file=sys.stderr)
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
