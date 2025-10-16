import platform
import time

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.custom_env import CustomEnv


def main(
    show_viewer: bool = True,
    device: str = "cpu",
) -> None:
    # Load checkpoint and env_args
    env_args = EnvArgsRegistry["custom_g1_mocap"]
    env = CustomEnv(args=env_args, num_envs=1, show_viewer=show_viewer, device=torch.device(device))  # type: ignore[arg-type]

    env.reset()

    def loop() -> None:
        env.scene.set_obj_pose(
            name="left_foot",
            pos=torch.tensor([[1.0, -0.1, 0.3]]),
            quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        env.scene.set_obj_pose(
            name="right_foot",
            pos=torch.tensor([[1.0, 0.1, 0.3]]),
            quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        env.scene.set_obj_pose(
            name="pelvis",
            pos=torch.tensor([[0.0, 0.0, 0.3]]),
            quat=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        )
        dof_pos = torch.zeros(env.robot.dof_dim, device=env.device)
        dof_pos[3] = 2
        env.set_dof_pos(dof_pos)
        left_foot_idx_local = env.robot.get_link_idx_local_by_name("left_ankle_roll_link")
        env.set_link_pose(
            link_idx_local=left_foot_idx_local,
            pos=torch.tensor([1.0, -0.1, 0.3]),
            quat=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        )
        while True:
            time.sleep(1)
            env.scene.step(update_visualizer=True, refresh_visualizer=False)

    try:
        if platform.system() == "Darwin":
            import threading

            threading.Thread(target=loop).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
