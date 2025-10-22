import argparse
import pickle
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
    # Create viewer env
    viewer_env_args = sim_env_registry["custom_g1_mocap"]
    assert isinstance(viewer_env_args, LeggedRobotEnvArgs)
    viewer_env = CustomEnv(
        args=viewer_env_args,
        num_envs=1,
        show_viewer=True,
    )

    # Parse save path
    save_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "robot_offset"
        / (args.save_config + ".yaml")
    )

    # Parse presample data path
    data_path = (
        Path(__file__).resolve().parent.parent / "config" / "robot_offset" / "collected_data.pkl"
    )

    collected_data: list[dict[str, Any]] = []
    if not args.presample:
        # Create OptiTrack env with zero offsets
        load_config_path = (
            Path(__file__).resolve().parent.parent / "config" / "optitrack_offset" / "default.yaml"
        )
        optitrack_env_args = real_env_registry["g1_links_tracking"].model_copy(
            update={"offset_config": load_config_path},
        )
        assert isinstance(optitrack_env_args, OptitrackEnvArgs)
        optitrack_env = OptitrackEnv(num_envs=1, args=optitrack_env_args)

        # Create low state handler
        real_env_args = sim_env_registry["g1_walk"]
        assert isinstance(real_env_args, LeggedRobotEnvArgs)
        real_env = UnitreeLeggedEnv(args=real_env_args, interactive=False)

        print(
            "[INFO] Starting collection... press 'c' to capture a sample, 'q' to stop and calibrate,"
        )
        print("       's' to save collected data and exit.")
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
                print(f"[INFO] Total samples: {len(collected_data)}, starting calibration...")
                break
            elif key == "s":
                print(f"[INFO] Saving {len(collected_data)} samples to {save_path} and exiting...")
                with open(data_path, "wb") as f:
                    pickle.dump(collected_data, f)
                return
    else:
        with open(data_path, "rb") as f:
            collected_data = pickle.load(f)
        print(f"[INFO] Loaded {len(collected_data)} samples, starting calibration...")

    # Saved for pre-commit, TODO
    def _touch(*args, **kwargs) -> None:
        pass

    _touch(G1_FK_TABLES, viewer_env, save_path)
    raise NotImplementedError("TODO")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_config", type=str, default="calibrated")
    parser.add_argument("--presample", action="store_true", default=False)
    # robot qpos calibration is irrelevant to mocap offsets
    # save to robot offset folder
    args = parser.parse_args()
    main(args)
