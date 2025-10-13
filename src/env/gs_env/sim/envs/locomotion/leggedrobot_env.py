import importlib
import platform
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

#
from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_euler,
)
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.leggedrobots import G1Robot
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")


class LeggedRobotEnv(BaseEnv):
    """
    Environment for Legged Robots.
    """

    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._refresh_visualizer = False if platform.system() == "Darwin" else True
        self._args = args
        self._eval_mode = eval_mode

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
            env_spacing=(1.0, 1.0),
        )

        # == setup the robot ==
        self._robot = G1Robot(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self._device,
        )

        # == set up camera ==
        self._floating_camera = self._scene.scene.add_camera(
            res=(480, 480),
            fov=60,
            GUI=False,
        )

        # == build the scene ==
        self._scene.build()

        # == setup reward scalars and functions ==
        self.dt = self._scene.scene.dt * args.robot_args.decimation
        self._reward_functions = {}
        module_name = f"gs_env.common.rewards.{self._args.reward_term}_terms"
        module = importlib.import_module(module_name)
        for key in args.reward_args.keys():
            reward_func = getattr(module, key, None)
            if reward_func is None:
                raise ValueError(f"Reward {key} not found in rewards module.")
            self._reward_functions[key] = reward_func(scale=args.reward_args[key] * self.dt)

        # some auxiliary variables
        self._max_sim_time = 20.0  # seconds

        #
        self._init()
        self.reset()

    def _init(self) -> None:
        # domain randomization
        self._robot.post_build_init(eval_mode=self._eval_mode)

        # initialize buffers
        self._action_buf = torch.zeros(
            (self.num_envs, self.action_dim, self._args.action_latency + 1), device=self._device
        )
        self._action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self.last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self.last_last_action = torch.zeros((self.num_envs, self.action_dim), device=self._device)
        self.torque = torch.zeros((self.num_envs, self.action_dim), device=self._device)

        self.base_default_pos: torch.Tensor = self._robot.default_pos[None, :].repeat(
            self.num_envs, 1
        )
        self.base_default_quat: torch.Tensor = self._robot.default_quat[None, :].repeat(
            self.num_envs, 1
        )
        self.base_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self.base_euler = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)

        global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        self.global_gravity = global_gravity[None, :].repeat(self.num_envs, 1)
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )

        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)

        self.link_contact_forces = torch.zeros(
            (self.num_envs, self._robot.n_links, 3), device=self.device, dtype=torch.float32
        )
        self.link_positions = torch.zeros(
            (self.num_envs, self._robot.n_links, 3), device=self.device, dtype=torch.float32
        )
        self.link_quaternions = torch.zeros(
            (self.num_envs, self._robot.n_links, 4), device=self.device, dtype=torch.float32
        )
        self.link_velocities = torch.zeros(
            (self.num_envs, self._robot.n_links, 3), device=self.device, dtype=torch.float32
        )

        #
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)

        # specify the space attributes
        actor_obs_spaces = {}
        for obs_term in self._args.actor_obs_terms:
            assert hasattr(self, obs_term), (
                f"Observation term {obs_term} not found in the environment."
            )
            actor_obs_spaces[obs_term] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(getattr(self, obs_term).shape[-1],),
                dtype=np.float32,
            )
        self._actor_observation_space = gym.spaces.Dict(actor_obs_spaces)
        critic_obs_spaces = {}
        for obs_term in self._args.critic_obs_terms:
            assert hasattr(self, obs_term), (
                f"Observation term {obs_term} not found in the environment."
            )
            critic_obs_spaces[obs_term] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(getattr(self, obs_term).shape[-1],),
                dtype=np.float32,
            )
        self._critic_observation_space = gym.spaces.Dict(critic_obs_spaces)
        self._info_space = gym.spaces.Dict({})
        self._extra_info = {}

        # rendering
        self._rendered_images = []
        self._rendering = False
        self.camera_lookat = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self.camera_pos = torch.tensor([-1.0, -1.0, 0.5], device=self._device)

        # Additional timers specific to this environment
        self._random_push_time = 4.0  # seconds
        self.time_since_random_push = torch.remainder(
            torch.arange(self.num_envs, device=self._device, dtype=torch.float32) * self.dt,
            self._random_push_time,
        )

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        default_pos = self._robot.default_pos[None, :].repeat(len(envs_idx), 1)
        default_quat = self._robot.default_quat[None, :].repeat(len(envs_idx), 1)
        default_dof_pos = self._robot.default_dof_pos[None, :].repeat(len(envs_idx), 1)
        random_euler = torch.zeros((len(envs_idx), 3), device=self._device)
        random_euler[:, :2] = (torch.rand(len(envs_idx), 2, device=self._device) - 0.5) * 0.3
        random_euler[:, 2] = torch.rand(len(envs_idx), device=self._device) * 2 * np.pi - np.pi
        random_dof_pos = torch.rand(len(envs_idx), self._robot.dof_dim, device=self._device) - 0.5
        random_dof_pos *= 0.3
        if self._eval_mode:
            random_euler *= 0
            random_dof_pos *= 0
        quat = quat_from_euler(random_euler)
        quat = quat_mul(quat, default_quat)
        dof_pos = default_dof_pos + random_dof_pos
        self.time_since_reset[envs_idx] = 0.0
        self._robot.set_state(pos=default_pos, quat=quat, dof_pos=dof_pos, envs_idx=envs_idx)

    def get_terminated(self) -> torch.Tensor:
        reset_buf = self.get_truncated()
        tilt_mask = torch.logical_or(
            torch.abs(self.base_euler[:, 0]) > 0.5,
            torch.abs(self.base_euler[:, 1]) > 0.5,
        )
        height_mask = self.base_pos[:, 2] < 0.5
        reset_buf |= tilt_mask
        reset_buf |= height_mask
        self.reset_buf[:] = reset_buf
        termination_dict = {}
        termination_dict["tilt"] = tilt_mask.clone()
        termination_dict["base_height"] = height_mask.clone()
        termination_dict["any"] = reset_buf.clone()
        self._extra_info["termination"] = termination_dict
        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        if self._eval_mode:
            self._max_sim_time = float("inf")
        time_out_buf = self.time_since_reset > self._max_sim_time
        self.time_out_buf[:] = time_out_buf
        return time_out_buf

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_buffers()
        obs_components = []
        for key in self._args.actor_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_noise = torch.randn_like(obs_gt) * self._args.obs_noises.get(key, 0.0)
            if self._eval_mode:
                obs_noise *= 0
            obs_components.append(obs_gt + obs_noise)
        actor_obs = torch.cat(obs_components, dim=-1)
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)
        return actor_obs, critic_obs

    def apply_action(self, action: torch.Tensor) -> None:
        action = action.detach().to(self._device)
        self._action = action
        self._action_buf[:] = torch.cat([self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1)
        exec_action = self._action_buf[:, :, 0]
        exec_action *= self._args.robot_args.action_scale

        self.torque *= 0

        # Apply actions and simulate physics
        for _ in range(self._args.robot_args.decimation):
            self.time_since_reset += self._scene.scene.dt
            self.time_since_random_push += self._scene.scene.dt

            self._robot.apply_action(action=exec_action)
            self._scene.scene.step(refresh_visualizer=self._refresh_visualizer)
            self.torque = torch.max(self.torque, torch.abs(self._robot.torque))

        self._update_buffers()

        # Render if rendering is enabled
        self._render_headless()

    def update_history(self) -> None:
        # save for reward computation
        self.last_last_action = self.last_action.clone()
        self.last_action = self._action.clone()

        push_env_ids = torch.nonzero(
            self.time_since_random_push >= self._random_push_time, as_tuple=False
        ).squeeze(-1)
        if not self._eval_mode:
            self._random_push(envs_idx=push_env_ids)
        self.time_since_random_push[push_env_ids] = 0.0

    def get_extra_infos(self) -> dict[str, Any]:
        self._update_buffers()
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        obs_tensor = torch.cat(obs_components, dim=-1)
        self._extra_info["observations"] = {"critic": obs_tensor}
        self._extra_info["time_outs"] = self.time_out_buf.clone()[:, None]
        return self._extra_info

    def _update_buffers(self) -> None:
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        base_quat_rel = quat_mul(self._robot.base_quat, quat_inv(self.base_default_quat))
        self.base_euler[:] = quat_to_euler(base_quat_rel)
        self.projected_gravity[:] = quat_apply(quat_inv(self.base_quat), self.global_gravity)
        inv_quat_yaw = quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self.device, dtype=torch.float)
        )
        self.base_lin_vel[:] = quat_apply(inv_quat_yaw, self._robot.get_vel())
        self.base_ang_vel[:] = quat_apply(quat_inv(base_quat_rel), self._robot.get_ang())

        self.link_contact_forces[:] = self._robot.link_contact_forces
        self.link_positions[:] = self._robot.link_positions
        self.link_quaternions[:] = self._robot.link_quaternions
        self.link_velocities[:] = self._robot.link_velocities

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_total_pos = torch.zeros(self.num_envs, device=self._device)
        reward_total_neg = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}
        for key, func in self._reward_functions.items():
            reward = func(
                {
                    "action": self._action,
                    "last_action": self.last_action,
                    "last_last_action": self.last_last_action,
                    "base_pos": self.base_pos,
                    "lin_vel": self.base_lin_vel,
                    "ang_vel": self.base_ang_vel,
                    "dof_pos": self.dof_pos,
                    "dof_vel": self.dof_vel,
                    "projected_gravity": self.projected_gravity,
                    "torque": self.torque,
                    "dof_pos_limits": self._robot.dof_pos_limits,
                    "commands": self.commands,
                }
            )
            if reward.sum() >= 0:
                reward_total_pos += reward
            else:
                reward_total_neg += reward
            reward_dict[f"{key}"] = reward.clone()
        reward_total = reward_total_pos * torch.exp(reward_total_neg)
        reward_dict["Total"] = reward_total
        reward_dict["TotalPositive"] = reward_total_pos
        reward_dict["TotalNegative"] = reward_total_neg

        return reward_total, reward_dict

    def _render_headless(self) -> None:
        if self._rendering and len(self._rendered_images) < 1000:
            robot_pos = self._robot.base_pos[0]
            robot_pos[2] = 0.7
            self._floating_camera.set_pose(
                pos=robot_pos + self.camera_pos,
                lookat=robot_pos + self.camera_lookat,
            )
            rgb, _, _, _ = self._floating_camera.render()
            self._rendered_images.append(rgb)

    def start_rendering(self) -> None:
        self._rendering = True
        self._rendered_images = []

    def stop_rendering(self, save_gif: bool = True, gif_path: str = ".") -> None:
        self._rendering = False
        if save_gif and self._rendered_images:
            self.save_gif(gif_path)

    def save_gif(self, gif_path: str, duration: int = 20) -> None:
        """
        Save the rendered images as a GIF.

        Args:
            gif_path: Path to save the GIF. If None, generates a timestamped filename.
            duration: Duration of each frame in milliseconds (default: 100ms)
        """
        if not self._rendered_images:
            print("No rendered images to save.")
            return

        # Convert numpy arrays to PIL Images
        pil_images = []
        for img_array in self._rendered_images:
            # Convert from numpy array to PIL Image
            # Assuming the image is in RGB format (H, W, 3)
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)

        # Save as GIF
        if pil_images:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=0,  # Infinite loop
            )
            print(f"GIF saved to: {gif_path}")
        else:
            print("No images to save as GIF.")

    def _random_push(self, envs_idx: torch.Tensor) -> None:
        if envs_idx.numel() > 0:
            # sample delta v in [-2, 2] for x and y
            delta_xy = torch.rand((len(envs_idx), 2), device=self._device) * 2.0 - 1.0
            cur_vel = self._robot.get_vel()[envs_idx]  # world-frame linear velocity (x,y,z)
            new_vel = cur_vel.clone()
            new_vel[:, :2] = new_vel[:, :2] + delta_xy
            self._robot.set_dofs_velocity(new_vel, envs_idx=envs_idx, dofs_idx_local=[0, 1, 2])

    def _random_push(self, envs_idx: torch.Tensor) -> None:
        if envs_idx.numel() > 0:
            # sample delta v in [-2, 2] for x and y
            delta_xy = torch.rand((len(envs_idx), 2), device=self._device) * 2.0 - 1.0
            cur_vel = self._robot.get_vel()[envs_idx]  # world-frame linear velocity (x,y,z)
            new_vel = cur_vel.clone()
            new_vel[:, :2] = new_vel[:, :2] + delta_xy
            self._robot.set_dofs_velocity(new_vel, envs_idx=envs_idx, dofs_idx_local=[0, 1, 2])

    def eval(self) -> None:
        self._eval_mode = True

    @property
    def scene(self) -> FlatScene:
        return self._scene

    @property
    def robot(self) -> G1Robot:
        return self._robot

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._robot.action_space

    @property
    def action_dim(self) -> int:
        act_dim = get_space_dim(self.action_space)
        return act_dim

    @property
    def action_scale(self) -> float:
        return self._args.robot_args.action_scale

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._actor_observation_space)

    @property
    def critic_obs_dim(self) -> int:
        return get_space_dim(self._critic_observation_space)

    @property
    def dof_pos(self) -> torch.Tensor:
        return self._robot.dof_pos

    @property
    def dof_vel(self) -> torch.Tensor:
        return self._robot.dof_vel

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
