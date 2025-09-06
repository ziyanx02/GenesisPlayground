from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch

#
from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.rewards import ActionL2Penalty, KeypointsAlign
from gs_env.common.utils.math_utils import quat_mul, quat_inv, quat_to_euler, quat_from_angle_axis
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
        self.rwd_action_l2 = ActionL2Penalty(scale=args.reward_args["rew_actions"] * dt)
        self.rwd_keypoints = KeypointsAlign(scale=args.reward_args["rew_keypoints"] * dt)

        # some auxiliary variables
        self._max_sim_time = 20.0  # seconds
        #
        self._init()
        self.reset()

    def _init(self):
        # specify the space attributes
        self._action_space = self._robot.action_space
        self._observation_space = gym.spaces.Dict(
            {
                "pose_vec": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "ee_quat": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "ref_position": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                ),  # 3D position
                "ref_quat": gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=np.float32
                ),  # 4D quaternion
                # "img": spaces.Box(
                #     low=0.0, high=255.0, shape=self._args.img_resolution, dtype=NP_SCALAR
                # ),  # RGB image
            }
        )
        self._info_space = gym.spaces.Dict({})

        # initialize buffers
        self._action_buf = torch.zeros(
            (self.num_envs, self.action_dim, self._args.action_latency + 1),
            device=self._device
        )
        self.base_default_pos: torch.Tensor = self._robot._default_pos[None, :].repeat(self.num_envs, 1)
        self.base_default_quat: torch.Tensor = self._robot._default_quat[None, :].repeat(self.num_envs, 1)
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self.base_euler = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)

        #
        self.goal_pose = torch.zeros(self.num_envs, 7, dtype=torch.float32, device=self._device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)
        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self._device, unit_length=0.5
        )

    def reset_idx(self, envs_idx: torch.IntTensor):
        self._robot.reset(envs_idx=envs_idx)
        # Generate random goal positions
        num_reset = len(envs_idx)
        # Random position within reachable workspace
        random_x = torch.rand(num_reset, device=self._device) * 0.3 + 0.15  # 0.15 to 0.45
        random_y = (torch.rand(num_reset, device=self._device) - 0.5) * 0.4  # -0.2 to 0.2
        random_z = torch.rand(num_reset, device=self._device) * 0.2 + 0.1  # 0.1 to 0.3

        self.goal_pose[envs_idx, 0] = random_x
        self.goal_pose[envs_idx, 1] = random_y
        self.goal_pose[envs_idx, 2] = random_z

        self.goal_pose[envs_idx] = torch.tensor(
            [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self._device
        )
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
        self.goal_pose[envs_idx, 3:7] = quat_mul(q_yaw, q_down)

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
        # Current end-effector pose
        ee_pos, ee_quat = self._robot.ee_pose[:, :3], self._robot.ee_pose[:, 3:7]
        #
        pos_diff = ee_pos - self.goal_pose[:, :3]
        obs_components = [
            pos_diff,  # 3D position difference
            ee_quat,  # current orientation (4D quaternion)
            self.goal_pose,  # goal pose (7D: pos + quat)
        ]
        obs_tensor = torch.cat(obs_components, dim=-1)

        extra_info = {
            "observations": {"critic": obs_tensor},
            "time_outs": self.reset_buf,
        }
        return obs_tensor

    def apply_action(self, action: torch.Tensor):
        action = action.detach().to(self._device)
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

    def _update_buffers(self):
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        base_quat_rel = quat_mul(self._robot.base_quat, quat_inv(self.base_default_quat))
        self.base_euler[:] = quat_to_euler(base_quat_rel)
        
        
        inv_quat_yaw = quat_from_angle_axis(-self.base_euler[:, 2],
                                            torch.tensor([0, 0, 1], device=self.device, dtype=torch.float))

    def get_info(self, envs_idx: torch.IntTensor | None = None) -> dict:
        if envs_idx is None:
            envs_idx = torch.IntTensor(range(self.num_envs))
        return dict()

    def compute_reward(self):
        # Squared-L2 action penalty
        reward_actions = self.rwd_action_l2(
            {
                "action": self.action_buf,  #
            }
        )
        # Key-point alignment
        reward_keypoints = self.rwd_keypoints(
            {
                "pose_a": self._robot.ee_pose,  # current pose
                "pose_b": self.goal_pose,  # goal pose
                "key_offsets": self.keypoints_offset,
            }
        )
        #
        reward_total = reward_actions + reward_keypoints

        reward_dict = {
            "reward_actions": reward_actions,
            "reward_keypoints": reward_keypoints,
            "reward_total": reward_total,
        }
        return reward_total, reward_dict

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @staticmethod
    def get_keypoint_offsets(batch_size, device, unit_length=0.5):
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at 0.
        """
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=TORCH_SCALAR,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    @property
    def device(self):
        return self._device

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

    @property
    def depth_shape(self):
        if self._camera is None and self._args.img_resolution is None:
            return None
        return (1, *self._args.img_resolution)

    @property
    def rgb_shape(self):
        if self._camera is None and self._args.img_resolution is None:
            return None
        return (4, *self._args.img_resolution)

    @property
    def img_resolution(self):
        if self._camera is None and self._args.img_resolution is None:
            return None
        return self._args.img_resolution
