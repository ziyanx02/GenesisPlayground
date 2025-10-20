import argparse
import sys
import termios
import tty
from pathlib import Path
from typing import Any

import numpy as np
from gs_env.real.config.registry import EnvArgsRegistry as real_env_registry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.real.optitrack_env import OptitrackEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.custom_env import CustomEnv

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
    real_env_args = sim_env_registry["g1_walk"]
    assert isinstance(real_env_args, LeggedRobotEnvArgs)
    real_env = UnitreeLeggedEnv(args=real_env_args, interactive=False)

    # Parse save path
    save_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "robot_offset"
        / (args.save_config + ".yaml")
    )

    print("[INFO] Starting collection... press 'c' to capture a sample, 'q' to quit and calibrate.")
    collected_data: list[dict[str, Any]] = []
    while True:
        key = getch()
        if key == "c":
            link_poses = optitrack_env.get_tracked_links(force_refresh=True)
            qpos = real_env.dof_pos[0].cpu().numpy().astype(np.float32)
            data = {
                "link_poses": link_poses,
                "qpos": qpos,
            }
            collected_data.append(data)
            print(f"[INFO] Sample #{len(collected_data)} collected.")
        elif key == "q":
            break

    print(f"[INFO] Total samples: {len(collected_data)}, starting calibration...")

    # Saved for pre-commit, TODO
    def _touch(*args, **kwargs) -> None:
        pass

    _touch(G1_FK_TABLES, viewer_env, save_path)
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_config", type=str, default="calibrated")
    # robot qpos calibration is irrelevant to mocap offsets
    # save to robot offset folder
    args = parser.parse_args()
    main(args)
