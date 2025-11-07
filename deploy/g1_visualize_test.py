import fire
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.custom_env import CustomEnv


def main(
    device: str = "cpu",
) -> None:
    device = "cpu" if not torch.cuda.is_available() else device
    device = torch.device(device)  # type: ignore[arg-type]

    env_args = EnvArgsRegistry["custom_g1_mocap"]
    sim_env = CustomEnv(args=env_args, num_envs=1, show_viewer=False, device=device)  # type: ignore[arg-type]

    env_args = EnvArgsRegistry["g1_walk"]

    print("=" * 80)
    print("Starting visualization")
    print(f"Device: {device}")
    print("=" * 80)

    try:
        for _ in range(1):
            link_idx_local = sim_env.get_link_idx_local_by_name("pelvis")
            sim_env.set_link_pose(
                link_idx_local,
                pos=torch.tensor([0.0, 0.0, 1.0], device=device),
                quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            )

            sim_env.set_dof_pos(torch.zeros(29, device=device))

            sim_env._update_buffers()  # noqa: SLF001

            dof_pos = sim_env.dof_pos[0].cpu().numpy()
            print("DOF Positions:", dof_pos)

            """
            desired left_ankle_pitch_link pose when all DOF positions are zero:
            [[-2.32404500e-06  1.18506454e-01 -7.39305735e-01 + 1.0]]
            """
            link_idx_local = sim_env.get_link_idx_local_by_name("left_ankle_pitch_link")
            link_pose = sim_env.get_link_pose(link_idx_local)[0].cpu().numpy()
            print("Link Pose:", link_pose)

            # sim_env.step_visualizer()

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
