import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gs_env.common.utils.math_utils import get_RT_between
from gs_env.real.config.registry import EnvArgsRegistry as real_env_registry
from gs_env.real.config.schema import OptitrackEnvArgs
from gs_env.real.optitrack_env import OptitrackEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry as sim_env_registry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.custom_env import CustomEnv

from .g1_r2s_config import G1_CB1_LINK_NAMES, G1_CB1_POS, G1_CB1_QUAT


def main(args: argparse.Namespace) -> None:
    # Create OptiTrack env with zero offsets
    offset_config_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "optitrack"
        / (args.offset_config + ".yaml")
    )
    optitrack_env_args = real_env_registry["g1_links_tracking"].model_copy(
        update={"offset_config": offset_config_path},
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

    # Set robot to initial pose
    pelvis_idx_local = viewer_env.robot.get_link_idx_local_by_name("pelvis")
    viewer_env.set_link_pose(
        pelvis_idx_local, pos=torch.tensor(G1_CB1_POS), quat=torch.tensor(G1_CB1_QUAT)
    )

    # Initialize zero offsets
    link_offsets = {}
    for name in G1_CB1_LINK_NAMES:
        link_offsets[name] = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "quat": np.array([1.0, 0.0, 0.0, 0.0]),
        }
    offset_sampled = 0

    def save_offsets(save_path: str) -> None:
        save_data = {}

        def represent_list(dumper: yaml.Dumper, data: list[Any]) -> yaml.nodes.SequenceNode:
            return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

        yaml.add_representer(list, represent_list)
        for name, offset in link_offsets.items():
            save_data[name] = {
                "pos": offset["pos"].tolist(),
                "quat": offset["quat"].tolist(),
            }
        with open(save_path, "w") as f:
            yaml.dump(save_data, f, sort_keys=False)
        print(f"Offsets saved to {save_path}.")

    save_path = (
        Path(__file__).resolve().parent.parent
        / "config"
        / "optitrack"
        / (args.offset_config + "_foot" + ".yaml")
    )

    while True:
        offset_sampled += 1
        frame = optitrack_env.get_tracked_links()
        for link_name in G1_CB1_LINK_NAMES:
            R_m, T_m = frame[link_name][1], frame[link_name][0]
            idx_local = viewer_env.robot.get_link_idx_local_by_name(link_name)
            Pose_s = viewer_env.get_link_pose(idx_local)
            R_s, T_s = Pose_s[1].cpu().numpy(), Pose_s[0].cpu().numpy()
            R_o, T_o = get_RT_between(R_m, T_m, R_s, T_s)
            m = 1 - 1 / offset_sampled
            link_offsets[link_name]["pos"] = m * link_offsets[link_name]["pos"] + (1 - m) * T_o
            link_offsets[link_name]["quat"] = m * link_offsets[link_name]["quat"] + (1 - m) * R_o

        if offset_sampled % 100 == 0:
            save_offsets(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offset_config", type=str, default="offset_default"
    )  # when calibrating foot, should always use offset_default
    args = parser.parse_args()
    main(args)
