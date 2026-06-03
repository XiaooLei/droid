from droid.controllers.oculus_controller import VRPolicy
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI
import argparse

parser = argparse.ArgumentParser(description="Launch DROID data collection GUI.")

parser.add_argument("--control-source", choices=["vr", "master"], default="vr", help="Teleoperation input source")
parser.add_argument("--master", action="store_true", help="Shortcut for --control-source master")
parser.add_argument("--left_controller", action="store_true", help="Use left oculus controller")
parser.add_argument("--right_controller", action="store_true", help="Use right oculus controller")
parser.add_argument(
    "--master-port",
    default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6023466-if00",
    help="DYNAMIXEL serial port for master arm",
)
parser.add_argument("--master-sample-dir", default=None, help="Path to dynamixel_sample_test; defaults to repo or /tmp")
parser.add_argument("--master-baudrate", type=int, default=1000000, help="DYNAMIXEL baudrate")
parser.add_argument("--master-frequency", type=float, default=50.0, help="DYNAMIXEL read frequency")
parser.add_argument("--master-ids", type=int, nargs="+", default=list(range(1, 9)), help="DYNAMIXEL IDs to read")
parser.add_argument("--master-joint-ids", type=int, nargs="+", default=None, help="DYNAMIXEL IDs for the 7 arm joints")
parser.add_argument("--master-gripper-id", type=int, default=8, help="DYNAMIXEL ID for gripper; set <=0 to disable")
parser.add_argument(
    "--master-joint-signs",
    type=float,
    nargs=7,
    default=[1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0],
    help="Sign correction for the 7 master joints",
)
parser.add_argument(
    "--master-joint-offsets",
    type=float,
    nargs=7,
    default=[0.0] * 7,
    help="Offset correction in radians for the 7 master joints",
)
parser.add_argument("--master-active-joints", type=int, nargs="+", default=list(range(1, 8)), help="1-based joints to control")
parser.add_argument("--master-joint-gain", type=float, default=0.9, help="Joint position scale for master arm control")
parser.add_argument("--master-max-target-delta", type=float, default=2.0, help="Max Franka target offset from origin in rad")
parser.add_argument("--master-gripper-mode", choices=["continuous", "calibrated"], default="calibrated")
parser.add_argument("--master-gripper-open", type=float, default=3.446855)
parser.add_argument("--master-gripper-closed", type=float, default=2.523398)
parser.add_argument("--master-gripper-open-command", type=float, default=0.0)
parser.add_argument("--master-gripper-closed-command", type=float, default=1.0)
parser.add_argument(
    "--master-sync-live-cameras",
    action="store_true",
    help="Read live camera images synchronously in the master-arm control loop.",
)


args = parser.parse_args()
if args.master:
    args.control_source = "master"

if args.control_source == "master":
    if not args.master_port:
        raise ValueError("--master-port is required when --control-source master")

    from droid.controllers.dynamixel_master_controller import DynamixelMasterPolicy

    env = RobotEnv(action_space="joint_position", gripper_action_space="position")
    gripper_id = None if args.master_gripper_id <= 0 else args.master_gripper_id
    controller = DynamixelMasterPolicy(
        port=args.master_port,
        ids=args.master_ids,
        joint_ids=args.master_joint_ids,
        gripper_id=gripper_id,
        sample_dir=args.master_sample_dir,
        baudrate=args.master_baudrate,
        frequency=args.master_frequency,
        joint_signs=args.master_joint_signs,
        joint_offsets=args.master_joint_offsets,
        active_joints=args.master_active_joints,
        joint_action_gain=args.master_joint_gain,
        max_target_delta=args.master_max_target_delta,
        gripper_mode=args.master_gripper_mode,
        gripper_master_open=args.master_gripper_open,
        gripper_master_closed=args.master_gripper_closed,
        gripper_open_command=args.master_gripper_open_command,
        gripper_closed_command=args.master_gripper_closed_command,
    )
    data_collector = DataCollecter(env=env, controller=controller)
    data_collector.control_source = "master"
    data_collector.read_cameras_during_control = args.master_sync_live_cameras
    user_interface = RobotGUI(robot=data_collector, right_controller=True, control_source="master")
else:
    env = RobotEnv()
    right_controller = not args.left_controller
    controller = VRPolicy(right_controller=right_controller)
    data_collector = DataCollecter(env=env, controller=controller)
    data_collector.control_source = "vr"
    user_interface = RobotGUI(robot=data_collector, right_controller=right_controller, control_source="vr")
