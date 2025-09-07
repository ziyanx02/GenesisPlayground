from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

#
from gs_env.common.bases.base_env import BaseEnv
import gs_env.common.rewards as rewards
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_inv,
    quat_mul,
    quat_from_angle_axis,
    quat_to_euler
)
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.leggedrobots import G1Robot
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")


class WalkingEnv(BaseEnv):
    """
    Walking Environment for Legged Robots.
    """

    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
    ):
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
        for key in args.reward_args.keys():
            if key not in rewards.__all__:
                raise ValueError(f"Reward {key} not found in rewards module.")
            self._reward_functions[key] = getattr(rewards, key)(scale=args.reward_args[key] * dt)

        # some auxiliary variables
        self._max_sim_time = 20.0  # seconds
        #
        self._init()
        self.reset()

    def _init(self):

        # domain randomization
        self._robot._post_build_init()

        # specify the space attributes
        self._action_space = self._robot.action_space
        self._observation_space = gym.spaces.Dict(
            {
                "last_action": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot._dof_dim,), dtype=np.float32),
                "dof_pos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot._dof_dim,), dtype=np.float32),
                "dof_vel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._robot._dof_dim,), dtype=np.float32),
                "projected_gravity": gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
                # "img": spaces.Box(
                #     low=0.0, high=255.0, shape=self._args.img_resolution, dtype=NP_SCALAR
                # ),  # RGB image
            }
        )
        self._info_space = gym.spaces.Dict({})
        self._extra_info = {}

        # initialize buffers
        self._action_buf = torch.zeros(
            (self.num_envs, self.action_dim, self._args.action_latency + 1),
            device=self._device
        )
        self._action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self._last_last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self.base_default_pos: torch.Tensor = self._robot._default_pos[None, :].repeat(self.num_envs, 1)
        self.base_default_quat: torch.Tensor = self._robot._default_quat[None, :].repeat(self.num_envs, 1)
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self.base_euler = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        self.global_gravity = global_gravity[None, :].repeat(self.num_envs, 1)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        #
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)

    def reset_idx(self, envs_idx: torch.IntTensor):
        self._robot.reset(envs_idx=envs_idx)
        # Generate random goal positions
        num_reset = len(envs_idx)
        # Random position within reachable workspace
        random_x = torch.rand(num_reset, device=self._device) * 0.3 + 0.15  # 0.15 to 0.45
        random_y = (torch.rand(num_reset, device=self._device) - 0.5) * 0.4  # -0.2 to 0.2
        random_z = torch.rand(num_reset, device=self._device) * 0.2 + 0.1  # 0.1 to 0.3

        q_down = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=self._device).repeat(
            num_reset, 1
        )
        random_yaw = torch.rand(num_reset, device=self._device) * 2 * np.pi - np.pi  # -pi to pi
        random_yaw *= 0.25  # reduce the range to [-pi/4, pi/4]
        # random_yaw *= 0.0  # reduce the range to [-pi/4, pi/4]
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self._device),
                torch.zeros(num_reset, device=self._device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )

        #
        self.time_since_reset[envs_idx] = 0.0

    def get_terminated(self) -> torch.Tensor:
        reset_buf = self.get_truncated()
        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        time_out_buf = self.time_since_reset > self._max_sim_time
        return time_out_buf

    def get_observations(self):
        self._update_buffers()
        # Prepare observation components
        obs_components = [
            last_action := self._last_action,  # last action
            dof_pos := self._robot.dof_pos,  # joint positions
            dof_vel := self._robot.dof_vel,  # joint velocities
            projected_gravity := self.projected_gravity,  # projected gravity in base frame
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)

        self._extra_info = {
            "observations": {"critic": obs_tensor},
            "time_outs": self.reset_buf,
        }
        return obs_tensor

    def apply_action(self, action: torch.Tensor):
        action = action.detach().to(self._device)
        self._action = action
        self._action_buf[:] = torch.cat(
            [self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1
        )
        exec_action = self._action_buf[:, :, 0]
        exec_action *= self._args.robot_args.action_scale

        # Apply actions and simulate physics
        for _ in range(self._args.robot_args.decimation):
            self.time_since_reset += self._scene.scene.dt
            self._robot.apply_action(action=exec_action)
            self._scene.scene.step()

        # save for reward computation
        self._last_last_action.copy_(self._last_action)
        self._last_action.copy_(self._action)

    def get_extra_infos(self) -> dict[str, Any]:
        return self._extra_info

    def _update_buffers(self):
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        base_quat_rel = quat_mul(self._robot.base_quat, quat_inv(self.base_default_quat))
        self.base_euler[:] = quat_to_euler(base_quat_rel)
        self.projected_gravity[:] = quat_apply(quat_inv(self.base_quat), self.global_gravity)
        inv_quat_yaw = quat_from_angle_axis(-self.base_euler[:, 2],
                                            torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

    def get_info(self, envs_idx: torch.IntTensor | None = None) -> dict:
        if envs_idx is None:
            envs_idx = torch.IntTensor(range(self.num_envs))
        return dict()

    def get_reward(self):
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}
        for key, func in self._reward_functions.items():
            reward = func(
                {
                    "action": self._action,
                    "last_action": self._last_action,
                    "last_last_action": self._last_last_action,
                }
            )
            reward_total += reward
            reward_dict[f"reward_{key}"] = reward

        return reward_total, reward_dict

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_dim(self):
        act_dim = get_space_dim(self._action_space) - 1  # -1 for the gripper actio
        return act_dim

    @property
    def actor_obs_dim(self):
        return get_space_dim(self._observation_space)

    @property
    def critic_obs_dim(self):
        num_critic_obs = get_space_dim(self._observation_space)
        return num_critic_obs

    # @property
    # def depth_shape(self):
    #     if self._camera is None and self._args.img_resolution is None:
    #         return None
    #     return (1, *self._args.img_resolution)

    # @property
    # def rgb_shape(self):
    #     if self._camera is None and self._args.img_resolution is None:
    #         return None
    #     return (4, *self._args.img_resolution)

    # @property
    # def img_resolution(self):
    #     if self._camera is None and self._args.img_resolution is None:
    #         return None
    #     return self._args.img_resolution
