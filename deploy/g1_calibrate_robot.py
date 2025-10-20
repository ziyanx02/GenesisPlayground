import argparse
import sys
import termios
import tty
from pathlib import Path
from typing import Any

import numpy as np
from gs_env.real.config.registry import EnvArgsRegistry as real_env_registry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.optitrack_env import OptitrackEnv
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.custom_env import CustomEnv
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.robots.config.schema import HumanoidRobotArgs

from .g1_r2s_config import G1_FK_TABLES


def getch() -> str:
    """Non-blocking single-key input (Linux/macOS)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main(args: argparse.Namespace) -> None:
    # Create OptiTrack env with zero offsets
    load_config_path = (
        Path(__file__).resolve().parent.parent / "config" / "optitrack_offset" / "default.yaml"
    )
    optitrack_env_args = real_env_registry["g1_links_tracking"].model_copy(
        update={"offset_config": load_config_path},
    )
    assert isinstance(optitrack_env_args, OptitrackEnvArgs)
    optitrack_env = OptitrackEnv(num_envs=1, args=optitrack_env_args)

    # Create viewer env
    viewer_env_args = sim_env_registry["custom_scene_g1_links_tracking"]
    assert isinstance(viewer_env_args, LeggedRobotEnvArgs)
    viewer_env = CustomEnv(
        args=viewer_env_args,
        num_envs=1,
        show_viewer=True,
    )

    # Create low state handler
    low_state_handler_robot_args = RobotArgsRegistry["g1_default"]
    assert isinstance(low_state_handler_robot_args, HumanoidRobotArgs)
    low_state_handler = LowStateMsgHandler(low_state_handler_robot_args)
    low_state_handler.init()

    save_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "robot_offset"
        / (args.save_config + ".yaml")
    )

    collected_data: list[dict[str, Any]] = []
    print("[INFO] Collection started. Press 'c' to capture a sample, 'q' to quit and calibrate.")
    while True:
        key = getch()
        if key == "c":
            link_poses = optitrack_env.get_tracked_links(force_refresh=True)
            qpos = low_state_handler.joint_pos.astype(np.float32)
            data = {
                "link_poses": link_poses,
                "qpos": qpos,
            }
            collected_data.append(data)
            print(f"[INFO] Sample #{len(collected_data)} collected.")
        elif key == "q":
            break

    print(f"[INFO] Total samples: {len(collected_data)}, starting calibration...")
    dof_names = low_state_handler_robot_args.dof_names
    dof_idx_local = [
        viewer_env.robot.get_joint_dofs_idx_local_by_name(name)[0] for name in dof_names
    ]

    # Saved for pre-commit, TODO
    def _touch(*args, **kwargs) -> None:
        pass

    _touch(G1_FK_TABLES, viewer_env, save_path, dof_idx_local)
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_config", type=str, default="calibrated")
    # robot qpos calibration is irrelevant to mocap offsets
    # save to robot offset folder
    args = parser.parse_args()
    main(args)
