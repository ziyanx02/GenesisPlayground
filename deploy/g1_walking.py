import time

import numpy as np
import torch
from gs_env.real.low_state_controller import LowStateCmdHandler
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from transforms3d import quaternions

task_name = "g1-walking"
ckpt_path = f"./ckpts/{task_name}.pt"

# base_init_quat = torch.tensor(cfg["environment"]["base_init_quat"])


def gs_transform_by_quat(pos, quat):
    qw, qx, qy, qz = quat.unbind(-1)

    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)

    return rotated_pos


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = EnvArgsRegistry["walk_default"]
    handler = LowStateCmdHandler(cfg)
    handler.init()
    handler.start()

    policy = torch.jit.load(ckpt_path)
    policy.to(device)
    policy.eval()
    cnt = 0

    default_dof_pos = handler.default_pos
    reset_dof_pos = handler.reset_pos.copy()
    commands = np.array(
        [
            0.0,
            0.0,
            0.0,
        ]
    )
    last_action = np.array([0.0] * len(cfg.robot_args.dof_names))
    try:
        while not handler.Start:
            time.sleep(0.1)

        print("Start runing policy")
        last_update_time = time.time()

        step_id = 0
        while not handler.emergency_stop:
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()
            projected_gravity = quaternions.rotate_vector(
                v=np.array([0, 0, -1]),
                q=quaternions.qinverse(handler.quat),
            )
            commands[0] = 1.0
            commands[1] = 0.0
            commands[2] = 0.0
            obs = np.concatenate(
                [
                    last_action,
                    # (np.array(handler.joint_pos) - default_dof_pos) * cfg["environment"]["observation"]["obs_scales"]["dof_pos"],
                    (np.array(handler.joint_pos) - default_dof_pos) * 1.0,
                    np.array(handler.joint_vel) * cfg.obs_scales["dof_vel"],
                    projected_gravity,
                    np.array(handler.ang_vel) * cfg.obs_scales["base_ang_vel"],
                    commands[:3],
                ]
            )
            print(obs)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t)[0].squeeze(0).detach().cpu().numpy().astype(np.float32)
            last_action = action
            # action[[5, 11]] = 0

            # action *= 0
            # dof_id = 1
            # action[dof_id] = np.sin(cnt / 100 * 3.14) * 4.0
            # print(action[dof_id])
            #### START FROM SMALL VALUES ####
            print(action)
            handler.target_pos = reset_dof_pos + 0.3 * (
                default_dof_pos + action * cfg.robot_args.action_scale - reset_dof_pos
            )
            step_id += 1
            cnt += 1
            # print(time.time() - last_update_time)
    except KeyboardInterrupt:
        pass

    handler.recover()
