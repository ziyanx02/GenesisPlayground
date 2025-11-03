import time
from pathlib import Path

import fire
import torch
import yaml
from gs_env.real.leggedrobot_env import UnitreeLeggedEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.custom_env import CustomEnv
from gs_env.sim.robots.config.schema import HumanoidRobotArgs


def main(
    device: torch.device | str = "cpu",
) -> None:
    device = "cpu" if not torch.cuda.is_available() else device
    device = torch.device(device)

    env_args = EnvArgsRegistry["custom_g1_mocap"]
    assert isinstance(env_args, LeggedRobotEnvArgs)
    sim_env = CustomEnv(args=env_args, num_envs=1, show_viewer=True, device=device)

    env_args = EnvArgsRegistry["g1_walk"]
    assert isinstance(env_args, LeggedRobotEnvArgs)
    real_env = UnitreeLeggedEnv(args=env_args, interactive=False, device=device)

    offset_path = (
        Path(__file__).resolve().parent.parent / "config" / "robot_offset" / "calibrated.yaml"
    )
    with open(offset_path) as f:
        offset_raw = yaml.safe_load(f)
    assert isinstance(env_args.robot_args, HumanoidRobotArgs)
    gs_joint_order = env_args.robot_args.dof_names
    qpos_offset = torch.tensor(
        [offset_raw[name] for name in gs_joint_order],
        dtype=torch.float32,
        device=device,
    )

    print("=" * 80)
    print("Starting visualization")
    print(f"Device: {device}")
    print("=" * 80)

    try:
        last_update_time = time.time()
        while True:
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.005)
                continue
            last_update_time = time.time()

            dof_pos = real_env.dof_pos[0] + qpos_offset

            sim_env.set_dof_pos(dof_pos)
            # sim_env.robot.set_state(quat=real_env.quat[0])
            link_idx_local = sim_env.get_link_idx_local_by_name("imu_in_torso")
            sim_env.set_link_pose(link_idx_local, quat=real_env.quat[0])
            sim_env.step_visualizer()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
