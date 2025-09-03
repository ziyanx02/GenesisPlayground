import genesis as gs
import numpy as np
import torch

#
from gs_env.common.bases import BaseGymEnv, spaces
from gs_env.common.rewards import ActionL2Penalty, KeypointsAlign
from gs_env.common.utils.gs_types import NP_SCALAR, TORCH_SCALAR
from gs_env.common.utils.math_utils import quat_mul
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import EnvArgs
from gs_env.sim.robots import PiperRobot
from gs_env.sim.scenes import FlatScene
from gs_env.sim.sensors.camera import Camera


class GoalReachingEnv(BaseGymEnv):
    """
    Goal Reaching Environment for Manipulators.
    """

    def __init__(
        self,
        args: EnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._args = args

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
        )

        # == setup the robot ==
        self._robot = PiperRobot(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self._device,
        )

        self._camera = None
        if self._args.img_resolution is not None:
            self._camera = Camera(scene=self._scene.scene)

        # == setup target entity
        self._target = self._scene.add_entity(
            gs.morphs.Box(size=(0.05, 0.05, 0.05), collision=False),
            material=gs.materials.Rigid(gravity_compensation=1),
        )
        # == build the scene ==
        self._scene.build()

        # == setup reward scalars and functions ==
        dt = self._scene.scene.dt
        self.rwd_action_l2 = ActionL2Penalty(scale=args.reward_args["rew_actions"] * dt)
        self.rwd_keypoints = KeypointsAlign(scale=args.reward_args["rew_keypoints"] * dt)

        # some auxiliary variables
        self._max_sim_time = 3.0  # seconds
        #
        self._init()
        self.reset()

    def _init(self) -> None:
        # specify the space attributes
        self._action_space = self._robot.action_space
        self._observation_space = spaces.Dict(
            {
                "pose_vec": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=NP_SCALAR),
                "ee_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=NP_SCALAR),
                "ref_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=NP_SCALAR
                ),  # 3D position
                "ref_quat": spaces.Box(
                    low=-1.0, high=1.0, shape=(4,), dtype=NP_SCALAR
                ),  # 4D quaternion
                # "img": spaces.Box(
                #     low=0.0, high=255.0, shape=self._args.img_resolution, dtype=NP_SCALAR
                # ),  # RGB image
            }
        )
        self._info_space = spaces.Dict({})

        #
        self.goal_pose = torch.zeros(self.num_envs, 7, dtype=TORCH_SCALAR, device=self._device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)
        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self._device, unit_length=0.5
        )
        self.action_buf = torch.zeros((self.num_envs, self.action_dim), device=self._device)

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        if envs_idx is None:
            envs_idx = torch.IntTensor(range(self.num_envs))
        self.reset_idx(envs_idx=envs_idx)

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
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
            [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=TORCH_SCALAR, device=self._device
        )
        q_down = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=TORCH_SCALAR, device=self._device).repeat(
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
        self._target.set_qpos(self.goal_pose[envs_idx], envs_idx=envs_idx)

    def is_episode_complete(self) -> torch.Tensor:
        self.time_out_buf = self.time_since_reset > self._max_sim_time

        # check if the ee is in the valid position
        self.reset_buf = self.time_out_buf

        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
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

        return obs_tensor, extra_info

    def get_ee_pose(self) -> torch.Tensor:
        return self._robot.ee_pose

    def get_obj_ori(self) -> torch.Tensor:
        return self.goal_pose[:, 3:7]

    def get_depth_image(self, normalize: bool = True) -> torch.Tensor:
        # Render depth image from the camera
        depth = self._camera.render()["depth"].permute(0, 3, 1, 2)  # shape (B, 1, H, W)
        if normalize:
            depth = torch.clamp(depth, min=0.0, max=10)
            depth = (depth - 0.0) / (10.0 - 0.0)  # normalize to [0, 1]
        return depth

    def get_rgb_image(self, normalize: bool = True) -> torch.Tensor:
        rgb = self._camera.render()["rgb"].permute(0, 3, 1, 2)  # shape (B, 4, H, W)
        if normalize:
            rgb = torch.clamp(rgb, min=0.0, max=255.0)
            rgb = (rgb - 0.0) / (255.0 - 0.0)
        return rgb

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = self.rescale_action(action)
        self.action_buf[:] = action.clone().to(self._device)

        # Apply actions and simulate physics
        self.time_since_reset += self._scene.scene.dt
        self._robot.apply_action(action=action)
        self._scene.scene.step()

        # Compute rewards
        reward_total, reward_dict = self.compute_reward()

        # Check if the episode is complete
        env_reset_ids = self.is_episode_complete()
        if len(env_reset_ids) > 0:
            self.reset_idx(env_reset_ids)

        #
        next_obs, extra_info = self.get_observations()
        extra_info["reward_dict"] = reward_dict
        return next_obs, reward_total.clone(), self.reset_buf, extra_info

    def get_info(
        self, envs_idx: torch.IntTensor | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        if envs_idx is None:
            envs_idx = torch.IntTensor(range(self.num_envs))
        return dict()

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        action_scale: torch.Tensor = torch.tensor(
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=TORCH_SCALAR, device=self._device
        )
        return action * action_scale

    def compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Squared-L2 action penalty
        reward_actions = self.rwd_action_l2(
            {
                "action": self.action_buf,  #
            }
        )
        # Key-point alignment
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
    def get_keypoint_offsets(
        batch_size: int, device: torch.device, unit_length: float = 0.5
    ) -> torch.Tensor:
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
    def device(self) -> torch.device:
        return self._device

    @property
    def action_dim(self) -> int:
        act_dim = get_space_dim(self._action_space) - 1  # -1 for the gripper actio
        return act_dim

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._observation_space)

    @property
    def critic_obs_dim(self) -> int:
        num_critic_obs = get_space_dim(self._observation_space)
        return num_critic_obs

    @property
    def depth_shape(self) -> tuple[int, int] | None:
        if self._camera is None and self._args.img_resolution is None:
            return None
        return (1, *self._args.img_resolution)

    @property
    def rgb_shape(self) -> tuple[int, int] | None:
        if self._camera is None and self._args.img_resolution is None:
            return None
        return (4, *self._args.img_resolution)

    @property
    def img_resolution(self) -> tuple[int, int] | None:
        if self._camera is None and self._args.img_resolution is None:
            return None
        return self._args.img_resolution
