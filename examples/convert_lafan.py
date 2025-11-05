import os
import pickle
import time
from pathlib import Path
from typing import Any, cast

import gs_env.sim.envs as gs_envs
import numpy as np
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.scenes.config.registry import SceneArgsRegistry


def lafan_to_motion_data(
    env: gs_envs.MotionEnv, data: torch.Tensor, show_viewer: bool = False
) -> None | dict[str, Any]:
    """
    Convert LAFAN data to motion data
    If show_viewer is True, the environment will be rendered in a separate thread.
    Returns:
        motion_data: The motion data
        None: If show_viewer is True, the environment will be rendered in a separate thread.
    """

    links = env.robot.robot.links
    link_names = [link.name for link in links]
    dof_names = env.dof_names

    motion_data = {}
    motion_data["fps"] = 30
    motion_data["link_names"] = link_names
    motion_data["dof_names"] = dof_names
    pos_list = []
    quat_list = []
    dof_pos_list = []
    link_pos_list = []
    link_quat_list = []

    def run() -> dict[str, Any]:
        nonlocal env, data, motion_data, show_viewer
        last_update_time = time.time()
        for i in range(len(data)):
            sliced_data = data[i]
            pos = sliced_data[:3]
            pos[2] -= 0.02
            quat = sliced_data[[6, 3, 4, 5]]
            dof_pos = sliced_data[7:]
            env.robot.set_state(pos=pos, quat=quat, dof_pos=dof_pos)
            env.update_buffers()
            pos_list.append(env.base_pos[0].clone())
            quat_list.append(env.base_quat[0].clone())
            dof_pos_list.append(env.dof_pos[0].clone())
            link_pos_list.append(env.link_positions[0].clone())
            link_quat_list.append(env.link_quaternions[0].clone())
            if show_viewer:
                env.scene.scene.step(refresh_visualizer=False)
                while time.time() - last_update_time < 1 / motion_data["fps"]:
                    time.sleep(0.01)
                last_update_time = time.time()

        motion_data["pos"] = torch.stack(pos_list).numpy()
        motion_data["quat"] = torch.stack(quat_list).numpy()
        motion_data["dof_pos"] = torch.stack(dof_pos_list).numpy()
        motion_data["link_pos"] = torch.stack(link_pos_list).numpy()
        motion_data["link_quat"] = torch.stack(link_quat_list).numpy()

        return motion_data

    try:
        if show_viewer:
            import threading

            threading.Thread(target=run).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            return run()
    except KeyboardInterrupt:
        return None


if __name__ == "__main__":
    show_viewer = False

    csv_files = "/Users/xiongziyan/Python/GenesisPlayground/assets/lafan/dance2_subject3.csv"
    # csv_files = "/Users/xiongziyan/Python/GenesisPlayground/assets/lafan/jumps1_subject1.csv"
    data = np.genfromtxt(csv_files, delimiter=",")
    data = torch.from_numpy(data).to(torch.float32)

    log_dir = Path("./assets/motion/lafan")
    os.makedirs(log_dir, exist_ok=True)
    motion_name = os.path.basename(csv_files).split(".")[0]
    motion_path = log_dir / (motion_name + ".pkl")

    env_args = cast(MotionEnvArgs, EnvArgsRegistry["g1_motion"]).model_copy(
        update={"scene_args": SceneArgsRegistry["flat_scene_legged"]}
    )
    env = gs_envs.MotionEnv(
        args=env_args,
        num_envs=1,
        show_viewer=show_viewer,
        device=torch.device("cpu"),
        eval_mode=True,
    )
    env.reset()

    motion_data = lafan_to_motion_data(env=env, data=data, show_viewer=show_viewer)
    if motion_data is not None:
        print(f"Saving motion data to {motion_path}")
        with open(motion_path, "wb") as f:
            pickle.dump(motion_data, f)
