# ROBOT SPECIFIC IMPORTS
import time
import threading
from typing import Optional

import numpy as np
import torch

# Franky imports
try:
    import franky
    HAS_FRANKY = True
except ImportError:
    HAS_FRANKY = False

# Gripper imports
try:
    import pyrobotiqgripper as rq
    HAS_ROBOTIQ = True
except ImportError:
    HAS_ROBOTIQ = False

from droid.misc.parameters import robot_ip
from droid.robot_ik.robot_ik_solver import RobotIKSolver

# UTILITY SPECIFIC IMPORTS
from droid.misc.transformations import add_poses, euler_to_quat, pose_diff, quat_to_euler
import os

class FrankyRobot:
    """Franky-based robot control for Franka robots."""

    ROBOT_IP = os.environ.get("ROBOT_IP", "172.16.0.3")  # FR3 Robot IP from env

    def __init__(self):
        if not HAS_FRANKY:
            raise ImportError("franky is not installed. Install with: pip install franky-control")

        self._ik_solver = RobotIKSolver()
        self._robot: Optional[franky.Robot] = None
        self._robot_lock = threading.RLock()
        self._last_robot_error = None
        self._robot_recover_count = 0
        self._robot_reconnect_count = 0
        self._gripper: Optional[object] = None
        self._control_thread: Optional[threading.Thread] = None
        self._stop_control = threading.Event()
        self._current_joint_positions: Optional[np.ndarray] = None
        self._joint_position_lock = threading.Lock()
        self._gripper_width = 0.0  # Current gripper position (0-1 normalized)
        self._max_gripper_width = 0.085  # Default for Robotiq 2F
        self._last_gripper_target = None
        self._gripper_deadband = float(os.environ.get("DROID_GRIPPER_DEADBAND", "0.01"))
        self._gripper_lock = threading.Lock()
        self._gripper_condition = threading.Condition(self._gripper_lock)
        self._gripper_worker_stop = threading.Event()
        self._gripper_worker_thread = None
        self._pending_gripper_target = None
        self._latest_requested_gripper_target = None
        self._gripper_busy = False
        self._gripper_submit_count = 0
        self._gripper_send_count = 0
        self._gripper_overwrite_count = 0
        self._gripper_error_count = 0

    def launch_controller(self):
        """No-op for compatibility."""
        pass

    def launch_robot(self):
        """Initialize robot and gripper connection via Franky."""
        with self._robot_lock:
            self._connect_robot_locked()

        # Gripper initialization using pyRobotiqGripper
        gripper_com_port = os.environ.get("GRIPPER_COM_PORT", "/dev/ttyUSB0")
        if self._gripper is None and HAS_ROBOTIQ and gripper_com_port:
            try:
                self._gripper = rq.RobotiqGripper(com_port=gripper_com_port)
                self._gripper.connect()
                self._gripper.activate()
                time.sleep(2)
                self._gripper.calibrate_speed()
                time.sleep(1)
                self._gripper_width = 1.0  # normalized position (1 = open)
                self._last_gripper_target = self._gripper_width
                self._latest_requested_gripper_target = self._gripper_width
                self._start_gripper_worker()
            except Exception as e:
                print(f"Warning: Gripper not available ({e}). Continuing without gripper.")
                self._gripper = None

        print(f"Franky robot initialized: IP={self.ROBOT_IP}")

    def establish_connection(self):
        """Compatibility RPC: reconnect to the robot without restarting the server."""
        self.launch_robot()

    def _connect_robot_locked(self, force=False):
        if self._robot is not None and not force:
            try:
                state = self._robot.state
                with self._joint_position_lock:
                    self._current_joint_positions = np.array(state.q)
                self._last_robot_error = None
                return
            except Exception as err:
                print(f"[FRANKY] Existing robot connection failed health check: {err}")
                force = True

        if force:
            self._robot_reconnect_count += 1
            self._robot = None

        self._robot = franky.Robot(self.ROBOT_IP)
        self._robot.relative_dynamics_factor = 0.1  # Start with 10% speed

        state = self._robot.state
        with self._joint_position_lock:
            self._current_joint_positions = np.array(state.q)

        self._controller_not_loaded = False
        self._last_robot_error = None

    def _recover_robot_locked(self, reason=None):
        """Recover Franka faults first, then reconnect if the object is stale."""
        self._last_robot_error = None if reason is None else str(reason)
        self._robot_recover_count += 1

        if self._robot is None:
            self._connect_robot_locked(force=True)
            return

        try:
            print(f"[FRANKY] Recovering robot after error: {reason}")
            self._robot.recover_from_errors()
            time.sleep(0.2)
            state = self._robot.state
            with self._joint_position_lock:
                self._current_joint_positions = np.array(state.q)
            self._last_robot_error = None
            return
        except Exception as recover_error:
            print(f"[FRANKY] recover_from_errors failed, reconnecting: {recover_error}")
            self._last_robot_error = str(recover_error)
            self._connect_robot_locked(force=True)

    def _run_robot_call(self, name, func, retries=1):
        last_error = None
        for attempt in range(retries + 1):
            with self._robot_lock:
                try:
                    self._connect_robot_locked()
                    return func()
                except Exception as err:
                    last_error = err
                    if attempt >= retries:
                        self._last_robot_error = str(err)
                        raise
                    self._recover_robot_locked(reason=f"{name}: {type(err).__name__}: {err}")
        raise last_error

    def recover_robot(self):
        """Manual RPC recovery hook for PC2/GUI without restarting the NUC process."""
        with self._robot_lock:
            self._recover_robot_locked(reason="manual recover_robot")
            state = self._robot.state
            return {
                "ok": True,
                "joint_positions": list(state.q),
                "recover_count": self._robot_recover_count,
                "reconnect_count": self._robot_reconnect_count,
            }

    def get_connection_status(self):
        return {
            "robot_connected": self._robot is not None,
            "last_robot_error": self._last_robot_error,
            "recover_count": self._robot_recover_count,
            "reconnect_count": self._robot_reconnect_count,
        }

    def kill_controller(self):
        """Stop any ongoing control."""
        self._stop_control.set()
        if self._control_thread and self._control_thread.is_alive():
            self._control_thread.join(timeout=2)
        self._stop_gripper_worker()

    def _start_gripper_worker(self):
        if self._gripper_worker_thread is not None and self._gripper_worker_thread.is_alive():
            return
        self._gripper_worker_stop.clear()
        self._gripper_worker_thread = threading.Thread(target=self._gripper_worker_loop, daemon=True)
        self._gripper_worker_thread.start()

    def _stop_gripper_worker(self):
        self._gripper_worker_stop.set()
        with self._gripper_condition:
            self._gripper_condition.notify_all()
        if self._gripper_worker_thread is not None:
            self._gripper_worker_thread.join(timeout=2)
            self._gripper_worker_thread = None

    def _gripper_worker_loop(self):
        while not self._gripper_worker_stop.is_set():
            with self._gripper_condition:
                while self._pending_gripper_target is None and not self._gripper_worker_stop.is_set():
                    self._gripper_condition.wait(timeout=0.1)
                if self._gripper_worker_stop.is_set():
                    break
                target_pos = self._pending_gripper_target
                self._pending_gripper_target = None
                self._gripper_busy = True

            target_bits = int(target_pos * 255)
            try:
                self._gripper.realTimeMove(target_bits)
                with self._gripper_condition:
                    self._gripper_width = target_pos
                    self._last_gripper_target = target_pos
                    self._gripper_send_count += 1
            except Exception as e:
                with self._gripper_condition:
                    self._gripper_error_count += 1
                print(f"Error updating gripper: {e}")
            finally:
                with self._gripper_condition:
                    self._gripper_busy = False

    def read_once(self):
        """Read robot state (simplified for Franky compatibility)."""
        def read():
            state = self._robot.state
            return {
                "q": list(state.q),
                "dq": list(state.dq),
                "O_T_EE": list(state.O_T_EE),
                "tau_J": list(state.tau_J) if hasattr(state, 'tau_J') else [0.0] * 7,
                "time": state.time if hasattr(state, 'time') else 0,
            }
        return self._run_robot_call("read_once", read, retries=1)

    def _enter_joint_control_mode(self):
        """Enter joint position control mode - must be called before motion."""
        def enter():
            try:
                self._robot.recover_from_errors()
            except Exception:
                pass

            self._robot.stop()

            for _ in range(40):  # Wait up to 2 seconds
                dq = np.abs(np.array(self._robot.state.dq))
                if dq.max() < 0.005:
                    break
                time.sleep(0.05)

            try:
                self._robot.move(franky.JointStopMotion())
            except Exception:
                pass

            time.sleep(1.0)

        return self._run_robot_call("_enter_joint_control_mode", enter, retries=1)

    def _move_to_joint_positions_once_locked(self, positions, speed_factor):
        self._robot.relative_dynamics_factor = speed_factor
        motion = franky.JointMotion(positions.tolist())
        self._robot.move(motion, asynchronous=True)
        return {
            "joint_positions": positions,
        }

    def move_to_joint_positions(self, positions, speed_factor=0.1, timeout=30):
        """Move to joint positions using Franky."""
        positions = np.array(positions)
        # Handle 8-element action (7 joints + 1 gripper)
        if len(positions) == 8:
            gripper_pos = positions[7]
            positions = positions[:7]
            self.update_gripper(gripper_pos, velocity=False)

        if len(positions) != 7:
            raise ValueError(f"Expected 7 joint positions, got {len(positions)}")

        try:
            self._run_robot_call(
                "move_to_joint_positions",
                lambda: self._move_to_joint_positions_once_locked(positions, speed_factor),
                retries=1,
            )
        except franky.ControlException as e:
            if "Reflex" in str(e):
                print(f"[FRANKY] Reflex detected, recovering...")
                try:
                    with self._robot_lock:
                        self._recover_robot_locked(reason=e)
                        self._move_to_joint_positions_once_locked(positions, speed_factor)
                    print(f"[FRANKY] Recovery successful.")
                except Exception as re:
                    print(f"[FRANKY] Recovery failed: {re}")
                    return False
            else:
                print(f"[FRANKY] ControlException (preempted, OK): {e}")
        except Exception as e:
            print(f"[FRANKY] move_to_joint_positions FAILED: {type(e).__name__}: {e}")
            return False

        # Update current positions
        with self._joint_position_lock:
            self._current_joint_positions = positions

        return True

    def update_joints(self, command, velocity=False, blocking=False, cartesian_noise=None):
        """Update joint positions."""
        command = np.array(command)

        if velocity:
            with self._joint_position_lock:
                current = np.array(self._current_joint_positions) if self._current_joint_positions is not None else np.zeros(7)
            joint_delta = self._ik_solver.joint_velocity_to_delta(command)
            target = (joint_delta + current).tolist()
        else:
            target = command.tolist()

        if blocking:
            self.move_to_joint_positions(target)
        else:
            thread = threading.Thread(target=self.move_to_joint_positions, args=(target,), daemon=True)
            thread.start()

    def update_gripper(self, command, velocity=True, blocking=False):
        """Submit a gripper target.

        Non-blocking calls use a single-slot latest-target worker so Robotiq
        stalls cannot block the arm control loop or build up stale commands.
        """
        if self._gripper is None:
            print("Warning: Gripper not initialized")
            return False

        with self._gripper_condition:
            current_gripper_width = self._latest_requested_gripper_target
            if current_gripper_width is None:
                current_gripper_width = self._gripper_width

        if velocity:
            gripper_delta = self._ik_solver.gripper_velocity_to_delta(command)
            target_pos = current_gripper_width + gripper_delta
            target_pos = float(np.clip(target_pos, 0, 1))
        else:
            # DROID convention: 0=open, 1=closed.
            target_pos = float(np.clip(command, 0, 1))

        with self._gripper_condition:
            reference_target = self._latest_requested_gripper_target
            if reference_target is None:
                reference_target = self._last_gripper_target
            if reference_target is not None and abs(target_pos - reference_target) < self._gripper_deadband:
                return False

            if blocking:
                self._pending_gripper_target = None
                self._latest_requested_gripper_target = target_pos
            else:
                if self._pending_gripper_target is not None:
                    self._gripper_overwrite_count += 1
                self._pending_gripper_target = target_pos
                self._latest_requested_gripper_target = target_pos
                self._gripper_submit_count += 1
                self._gripper_condition.notify()
                return True

        target_bits = int(target_pos * 255)
        try:
            self._gripper.realTimeMove(target_bits)
            with self._gripper_condition:
                self._gripper_width = target_pos
                self._last_gripper_target = target_pos
                self._gripper_send_count += 1
            return True
        except Exception as e:
            with self._gripper_condition:
                self._gripper_error_count += 1
            print(f"Error updating gripper: {e}")
            return False

    def get_gripper_async_status(self):
        with self._gripper_condition:
            return {
                "gripper_async_busy": self._gripper_busy,
                "gripper_async_has_pending": self._pending_gripper_target is not None,
                "gripper_async_submit_count": self._gripper_submit_count,
                "gripper_async_send_count": self._gripper_send_count,
                "gripper_async_overwrite_count": self._gripper_overwrite_count,
                "gripper_async_error_count": self._gripper_error_count,
            }

    def update_command(self, command, action_space="cartesian_velocity", gripper_action_space="velocity", blocking=False):
        """Main entry point for RobotEnv.update_robot() - maps action to robot command."""
        timing = {}
        start_time = time.perf_counter()
        action_dict = self.create_action_dict(
            action=np.array(command),
            action_space=action_space,
            gripper_action_space=gripper_action_space
        )
        timing["create_action_dict_ms"] = (time.perf_counter() - start_time) * 1000

        # Execute based on action_space type
        joint_start = time.perf_counter()
        if "cartesian" in action_space:
            joint_position = action_dict.get("joint_position")
            if joint_position is not None:
                self.update_joints(joint_position, velocity=False, blocking=blocking)
            else:
                print(f"[FRANKY] update_command: action_space={action_space}, joint_position=None (IK failed?)")
        elif "joint" in action_space:
            joint_position = action_dict.get("joint_position")
            if joint_position is not None:
                self.update_joints(joint_position, velocity=False, blocking=blocking)
            else:
                print(f"[FRANKY] update_command: action_space={action_space}, joint_position=None")
        timing["update_joints_ms"] = (time.perf_counter() - joint_start) * 1000

        # Update gripper
        gripper_start = time.perf_counter()
        gripper_command_sent = False
        gripper_pos = action_dict.get("gripper_position")
        if gripper_pos is not None:
            gripper_command_sent = self.update_gripper(gripper_pos, velocity=False, blocking=blocking)
        timing["update_gripper_ms"] = (time.perf_counter() - gripper_start) * 1000
        timing["gripper_command_sent"] = gripper_command_sent
        timing.update(self.get_gripper_async_status())
        timing["total_server_update_command_ms"] = (time.perf_counter() - start_time) * 1000
        action_dict["control_timing"] = timing

        # Convert numpy arrays to lists for zerorpc serialization
        return self._convert_action_dict(action_dict)

    def _convert_action_dict(self, action_dict):
        """Convert numpy arrays to lists for zerorpc serialization."""
        result = {}
        for key, value in action_dict.items():
            if key == "robot_state":
                result[key] = value  # robot_state is already a dict
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = list(value)
            else:
                result[key] = value
        return result

    def get_joint_positions(self):
        """Get current joint positions."""
        with self._joint_position_lock:
            if self._current_joint_positions is not None:
                return list(self._current_joint_positions)
        return self._run_robot_call("get_joint_positions", lambda: list(self._robot.state.q), retries=1)

    def get_joint_velocities(self):
        """Get current joint velocities."""
        return self._run_robot_call("get_joint_velocities", lambda: list(self._robot.state.dq), retries=1)

    def get_gripper_position(self):
        """Get gripper position (normalized 0-1, 1=open, 0=closed)."""
        if self._gripper is None:
            return self._gripper_width
        try:
            pos_bits = self._gripper.position()
            self._gripper_width = pos_bits / 255.0
            return float(np.clip(self._gripper_width, 0, 1))
        except Exception as e:
            print(f"Error getting gripper position: {e}")
            return self._gripper_width

    def get_ee_pose(self):
        """Get end-effector pose as [position, euler_angles]."""
        return self._run_robot_call("get_ee_pose", self._get_ee_pose_locked, retries=1)

    def _get_ee_pose_locked(self):
        cartesian_state = self._robot.current_cartesian_state
        # pose is RobotPose with end_effector_pose (Affine)
        ee_pose = cartesian_state.pose.end_effector_pose

        # Get position - translation can be numpy array
        position = np.array(ee_pose.translation)
        if position.ndim > 0:
            position = position.flatten()

        # Get rotation as quaternion and convert to euler
        quat = np.array(ee_pose.quaternion)
        if quat.ndim > 0:
            quat = quat.flatten()
        euler = self._quat_to_euler(quat.tolist())

        return (np.concatenate([position, euler])).tolist()

    def _quat_to_euler(self, quat):
        """Convert quaternion to Euler angles (XYZ convention)."""
        x, y, z, w = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def get_robot_state(self):
        """Get full robot state as a tuple (state_dict, timestamp_dict)."""
        def read_state():
            state = self._robot.state
            ee_pose = self._get_ee_pose_locked()
            gripper_position = self.get_gripper_position()

            state_dict = {
                "cartesian_position": ee_pose,
                "gripper_position": gripper_position,
                "joint_positions": list(state.q),
                "joint_velocities": list(state.dq),
                "joint_torques_computed": list(state.tau_J) if hasattr(state, 'tau_J') else [0.0] * 7,
                "prev_joint_torques_computed": list(state.tau_J) if hasattr(state, 'tau_J') else [0.0] * 7,
                "prev_joint_torques_computed_safened": list(state.tau_J) if hasattr(state, 'tau_J') else [0.0] * 7,
                "motor_torques_measured": list(state.tau_J) if hasattr(state, 'tau_J') else [0.0] * 7,
                "prev_controller_latency_ms": 0,
                "prev_command_successful": True,
            }

            timestamp_dict = {
                "robot_timestamp_seconds": int(state.time.to_sec()) if hasattr(state.time, 'to_sec') else 0,
                "robot_timestamp_nanos": 0,
            }

            return state_dict, timestamp_dict

        return self._run_robot_call("get_robot_state", read_state, retries=1)

    def is_running_policy(self):
        """Check if robot is currently moving."""
        return False

    def start_cartesian_impedance(self):
        """Start Cartesian impedance control mode. No-op in this version."""
        pass

    def update_desired_joint_positions(self, positions):
        """Update desired joint positions (non-blocking)."""
        positions_list = positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        thread = threading.Thread(target=self.move_to_joint_positions, args=(positions_list,))
        thread.start()

    def terminate_current_policy(self):
        """Terminate current motion."""
        self.kill_controller()

    def adaptive_time_to_go(self, desired_joint_position, t_min=0, t_max=4):
        """Calculate time to go for motion."""
        curr = np.array(self.get_joint_positions())
        displacement = np.abs(np.array(desired_joint_position) - curr).max()
        return min(t_max, max(t_min, displacement * 2))

    def create_action_dict(self, action, action_space, gripper_action_space=None, robot_state=None):
        """Create action dictionary from raw action."""
        action = np.array(action)
        if robot_state is None:
            robot_state = self.get_robot_state()[0]

        action_dict = {"robot_state": robot_state}
        velocity = "velocity" in action_space

        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"

        # Handle gripper action
        if gripper_action_space == "velocity":
            action_dict["gripper_velocity"] = float(action[-1])
            # action[-1] is trigger value (0-1): 1=fully pressed (close), 0=released (open)
            # Direct position mapping: trigger directly maps to target position
            trigger_value = float(action[-1])
            target_pos = trigger_value  # No inversion - press=close, release=open
            action_dict["gripper_position"] = float(np.clip(target_pos, 0, 1))
        else:
            action_dict["gripper_position"] = float(np.clip(action[-1], 0, 1))

        # Handle position/velocity control
        if "cartesian" in action_space:
            if velocity:
                action_dict["cartesian_velocity"] = action[:-1].tolist()
                cartesian_delta = self._ik_solver.cartesian_velocity_to_delta(action[:-1])
                action_dict["cartesian_position"] = add_poses(
                    cartesian_delta, robot_state["cartesian_position"]
                ).tolist()
                action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                    action_dict["cartesian_velocity"], robot_state=robot_state
                ).tolist()
                joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["cartesian_position"] = action[:-1].tolist()
                cartesian_delta = pose_diff(action[:-1], robot_state["cartesian_position"])
                cartesian_velocity = self._ik_solver.cartesian_delta_to_velocity(cartesian_delta)
                action_dict["joint_velocity"] = self._ik_solver.cartesian_velocity_to_joint_velocity(
                    cartesian_velocity, robot_state=robot_state
                ).tolist()
                joint_delta = self._ik_solver.joint_velocity_to_delta(action_dict["joint_velocity"])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()

        if "joint" in action_space:
            if velocity:
                action_dict["joint_velocity"] = action[:7].tolist()
                joint_delta = self._ik_solver.joint_velocity_to_delta(action[:7])
                action_dict["joint_position"] = (joint_delta + np.array(robot_state["joint_positions"])).tolist()
            else:
                action_dict["joint_position"] = action[:7].tolist()

        return action_dict
