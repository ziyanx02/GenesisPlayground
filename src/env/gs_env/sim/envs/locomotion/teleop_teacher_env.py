from typing import Any
import importlib

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_inv,
    quat_mul,
    quat_from_angle_axis,
    quat_to_euler,
    quat_from_euler,
)
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import TeleopTeacherEnvArgs
from gs_env.sim.robots.leggedrobots import G1Robot
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")

class TeleopTeacherEnv(BaseEnv):
    """
    TODO: Move into a single leggedgym env and control by config
    Teacher environment for motion imitation.
    """

    def __init__(
        self,
        args: TeleopTeacherEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._args = args

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))
        
        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
        )

        # == setup the robot ==
        self._robot = G1Robot(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self._device,
        )

        # == build the scene ==
        self._scene.build()

        # == setup reward scalars and functions ==
        dt = self._scene.scene.dt
        self._reward_functions = {}
        module_name = f"gs_env.common.rewards.{self._args.reward_term}_terms"
        module = importlib.import_module(module_name)
        for key in args.reward_args.keys():
            reward_func = getattr(module, key, None)
            if reward_func is None:
                raise ValueError(f"Reward {key} not found in rewards module.")
            self._reward_functions[key] = reward_func(scale=args.reward_args[key] * dt)

        # == auxiliary variables ==
        self._max_sim_time = 20.0  # seconds
        self._future_steps = self._args.future_steps
        self._common_step_counter = 0

        self._init()

        # == load trajectories ==
        self.load_trajectories()
        
        self.resample_trajectories_and_reset()

    def _init(self) -> None:

        # domain randomization
        self._robot.post_build_init()

        # specify the space attributes
        self._action_space = self._robot.action_space
        self._observation_space = gym.spaces.Dict(
            {
                "last_action": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.dof_dim,), dtype=np.float32),
                "pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "quat": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                "dof_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.dof_dim,), dtype=np.float32),
                "dof_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot.dof_dim,), dtype=np.float32),
                "projected_gravity": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                "ang_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "ref_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._future_steps * 3,), dtype=np.float32),
                "ref_quat": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._future_steps * 4, ), dtype=np.float32),
                "ref_dof_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._future_steps * self._robot.dof_dim, ), dtype=np.float32),
            }
        )
        self._info_space = gym.spaces.Dict({})
        self._extra_info = {}
        self._setup_buffers()
    
    def _setup_buffers(self) -> None:
        # == reference buffers ==
        self._ref_dof_pos = torch.zeros((self.num_envs, 1, self._robot.dof_dim), device=self._device)
        self._ref_pos = torch.zeros((self.num_envs, 1, 3), device=self._device)
        self._ref_quat = torch.zeros((self.num_envs, 1, 4), device=self._device)
        self._ref_start = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._ref_len = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # initialize buffers
        self._action_buf = torch.zeros(
            (self.num_envs, self.action_dim, self._args.action_latency + 1),
            device=self._device
        )
        self._action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._last_last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._torque = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self.base_euler = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        self.global_gravity = global_gravity[None, :].repeat(self.num_envs, 1)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        #
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)
    
    def load_trajectories(self) -> None:
        # data = np.load(self._args.motion_path, allow_pickle=True)
        data = {"tmp": {"dof_pos": np.zeros((100, 36)), "length": 100, "fps": 30}}
        # "name": {"dof_pos": ndarray(T, 36), "length": int, "fps": int}
        self.raw_motion = []
        self.raw_motion_lengths = []
        for arr in data.values():
            self.raw_motion_lengths.append(arr["length"])
            traj = torch.tensor(arr["dof_pos"], device=self._device, dtype=torch.float32)
            # pad trajectory with last frame future_steps times
            pad_len = self._future_steps
            if pad_len > 0:
                pad = traj[-1:].repeat(pad_len, 1)
                traj = torch.cat([traj, pad], dim=0)
            self.raw_motion.append(traj)

    def resample_trajectories_and_reset(self) -> None:
        # TODO: augmentation for trajectories
        # 1: package lost (freeze motion)
        # 2: random orientation
        data_len = len(self.raw_motion)
        sample_data_idx = torch.randint(0, data_len, (self.num_envs,), device=self._device)
        for i in range(self.num_envs):
            self._ref_len[i] = self.raw_motion_lengths[sample_data_idx[i]]
        max_len = int(torch.max(self._ref_len).item()) + self._future_steps
        self._ref_dof_pos = torch.zeros((self.num_envs, max_len, self._robot.dof_dim), device=self._device)
        self._ref_pos = torch.zeros((self.num_envs, max_len, 3), device=self._device)
        self._ref_quat = torch.zeros((self.num_envs, max_len, 4), device=self._device)
        for i in range(self.num_envs):
            self._ref_dof_pos[i, :self._ref_len[i]+self._future_steps] = self.raw_motion[sample_data_idx[i]][:, 7:]
            self._ref_pos[i, :self._ref_len[i]+self._future_steps] = self.raw_motion[sample_data_idx[i]][:, :3]
            self._ref_quat[i, :self._ref_len[i]+self._future_steps] = self.raw_motion[sample_data_idx[i]][:, 3:7]
        
        envs_idx = torch.IntTensor(range(self.num_envs))
        self.reset_idx(envs_idx=envs_idx)
    
    def resample_start_time(self, envs_idx: torch.IntTensor) -> None:
        for i in envs_idx:
            self._ref_start[i] = torch.randint(0, int(self._ref_len[i].item()), (1,), device=self._device)

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        # TODO: domain randomization
        # 1: initial pose noise
        self._reset_buffers(envs_idx=envs_idx)
        self.resample_start_time(envs_idx=envs_idx)
        
        default_pos = self._ref_pos[envs_idx, self._ref_start]
        default_quat = self._ref_quat[envs_idx, self._ref_start]
        default_dof_pos = self._ref_dof_pos[envs_idx, self._ref_start]
        dof_pos = (
            default_dof_pos
            + (torch.rand(len(envs_idx), self._robot.dof_dim, device=self._device) - 0.5) * 0.3
        )
        self._robot.set_state(pos=default_pos, quat=default_quat, dof_pos=dof_pos, envs_idx=envs_idx)

        self._episode_length[envs_idx] = 0

    def get_terminated(self) -> torch.Tensor:
        reset_buf = self.get_truncated()
        reset_buf |= self._ref_start[:] + self._episode_length[:] >= self._ref_len[:]
        reset_buf |= torch.logical_or(
            torch.abs(self.base_euler[:, 0]) > 0.3,
            torch.abs(self.base_euler[:, 1]) > 0.3,
        )
        reset_buf |= self.base_pos[:, 2] < 0.3
        self.reset_buf[:] = reset_buf
        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        time_out_buf = self.time_since_reset > self._max_sim_time
        self.time_out_buf[:] = time_out_buf
        return time_out_buf

    def get_observations(self) -> torch.Tensor:
        self._update_buffers()
        # Prepare observation components

        # Reference batch indices
        envs_idx = torch.arange(self.num_envs, device=self._device)
        time_steps = torch.arange(self._future_steps, device=self._device)
        ref_indices = self._ref_start.unsqueeze(1) + time_steps.unsqueeze(0)  # [num_envs, future_steps]

        obs_components = [
            last_action := self._last_action,  # last action
            pos := self.base_pos,  # base position
            quat := self.base_quat,  # base orientation
            dof_pos := self._robot.dof_pos,  # joint positions
            dof_vel := self._robot.dof_vel,  # joint velocities
            projected_gravity := self.projected_gravity,  # projected gravity in base frame
            ang_vel := self.base_ang_vel,  # angular velocity in base frame
            ref_pos := self._ref_pos[envs_idx.unsqueeze(1), ref_indices].view(self.num_envs, -1),
            ref_quat := self._ref_quat[envs_idx.unsqueeze(1), ref_indices].view(self.num_envs, -1),
            ref_dof_pos := self._ref_dof_pos[envs_idx.unsqueeze(1), ref_indices].view(self.num_envs, -1),
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)

        self._extra_info = {
            "observations": {"critic": obs_tensor},
            "time_outs": self.time_out_buf.clone(),
        }
        return obs_tensor

    def apply_action(self, action: torch.Tensor) -> None:
        action = action.detach().to(self._device)
        self._action = action
        self._action_buf[:] = torch.cat(
            [self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1
        )
        exec_action = self._action_buf[:, :, 0]
        exec_action *= self._args.robot_args.action_scale

        self._torque *= 0

        # Apply actions and simulate physics
        for _ in range(self._args.robot_args.decimation):
            self.time_since_reset += self._scene.scene.dt
            self._robot.apply_action(action=exec_action)
            self._scene.scene.step()
            self._torque = torch.max(self._torque, torch.abs(self._robot.torque))

        # save for reward computation
        self._last_last_action.copy_(self._last_action)
        self._last_action.copy_(self._action)

        # update step pointer
        self._episode_length += 1
        self._common_step_counter += 1

        # resample
        if self._common_step_counter % self._args.resample_interval == 0:
            self.resample_trajectories_and_reset()

    def get_extra_infos(self) -> dict[str, Any]:
        return self._extra_info
    
    def _reset_buffers(self, envs_idx: torch.IntTensor) -> None:
        self._action[envs_idx] = 0
        self._last_action[envs_idx] = 0
        self._last_last_action[envs_idx] = 0
        self._torque[envs_idx] = 0

    def _update_buffers(self) -> None:
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        base_quat_rel = self._robot.base_quat
        self.base_euler[:] = quat_to_euler(base_quat_rel)
        self.projected_gravity[:] = quat_apply(quat_inv(self.base_quat), self.global_gravity)
        inv_quat_yaw = quat_from_angle_axis(-self.base_euler[:, 2],
                                            torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))
        self.base_lin_vel[:] = quat_apply(inv_quat_yaw, self._robot.get_vel())
        self.base_ang_vel[:] = quat_apply(quat_inv(base_quat_rel), self._robot.get_ang())

    def get_info(self, envs_idx: torch.IntTensor | None = None) -> dict[str, Any]:
        if envs_idx is None:
            envs_idx = torch.IntTensor(range(self.num_envs))
        return dict()

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_total_pos = torch.zeros(self.num_envs, device=self._device)
        reward_total_neg = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}

        envs_idx = torch.arange(self.num_envs, device=self._device)
        ref_idx = self._ref_start + self._episode_length

        for key, func in self._reward_functions.items():
            reward = func(
                {
                    "action": self._action,
                    "last_action": self._last_action,
                    "last_last_action": self._last_last_action,
                    "base_pos": self.base_pos,
                    "lin_vel": self.base_lin_vel,
                    "ang_vel": self.base_ang_vel,
                    "dof_pos": self._robot.dof_pos,
                    "dof_vel": self._robot.dof_vel,
                    "projected_gravity": self.projected_gravity,
                    "torque": self._torque,
                    "dof_pos_limits": self._robot.dof_pos_limits,
                    "target_pos": self._ref_pos[envs_idx, ref_idx],
                    "target_quat": self._ref_quat[envs_idx, ref_idx],
                    "target_dof_pos": self._ref_dof_pos[envs_idx, ref_idx]
                }
            )
            if reward.sum() >= 0:
                reward_total_pos += reward
            else:
                reward_total_neg += reward
            reward_dict[f"reward_{key}"] = reward
        reward_total = reward_total_pos * torch.exp(reward_total_neg)
        reward_dict["reward_total"] = reward_total

        return reward_total, reward_dict

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_dim(self) -> int:
        act_dim = get_space_dim(self._action_space)
        return act_dim

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._observation_space)

    @property
    def critic_obs_dim(self) -> int:
        num_critic_obs = get_space_dim(self._observation_space)
        return num_critic_obs