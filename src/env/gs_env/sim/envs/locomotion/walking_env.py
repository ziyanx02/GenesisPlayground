import torch

#
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_inv,
)
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class WalkingEnv(LeggedRobotEnv):
    """
    Walking Environment for Legged Robots.
    """

    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        # Initialize base legged-robot environment
        super().__init__(
            args=args,
            num_envs=num_envs,
            show_viewer=show_viewer,
            device=device,
            eval_mode=eval_mode,
        )

    def _init(self) -> None:
        # Pre-parent: allocate feet-related buffers required by observation terms
        self.feet_height = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_z_velocity = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_orientation = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx), 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_contact_force = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_first_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self.device,
            dtype=torch.float32,
        )

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # Additional timers specific to this environment
        self._command_resample_time = 10.0  # seconds
        self.time_since_resample = torch.zeros(self.num_envs, device=self._device)

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        super().reset_idx(envs_idx=envs_idx)
        self.feet_air_time[envs_idx] = 0.0
        self._resample_commands(envs_idx=envs_idx)
        self.time_since_resample[envs_idx] = 0.0

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
            self.time_since_resample += self._scene.scene.dt
            self.time_since_random_push += self._scene.scene.dt

            self._robot.apply_action(action=exec_action)
            self._scene.scene.step(refresh_visualizer=self._refresh_visualizer)
            self.torque = torch.max(self.torque, torch.abs(self._robot.torque))

        self._update_buffers()

        # Render if rendering is enabled
        self._render_headless()
        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.feet_contact
        self.feet_air_time += self.dt

    def update_history(self) -> None:
        super().update_history()
        self.feet_air_time *= 1 - self.feet_contact

        resample_env_ids = torch.nonzero(
            self.time_since_resample > self._command_resample_time, as_tuple=False
        ).squeeze(-1)
        self._resample_commands(envs_idx=resample_env_ids)
        self.time_since_resample[resample_env_ids] = 0.0

    def _update_buffers(self) -> None:
        super()._update_buffers()
        self.feet_contact_force[:] = self.link_contact_forces[:, self._robot.foot_links_idx, 2]
        self.feet_contact[:] = self.feet_contact_force > 1.0
        self.feet_height[:] = self.link_positions[:, self._robot.foot_links_idx, 2]
        self.feet_z_velocity[:] = self.link_velocities[:, self._robot.foot_links_idx, 2]
        feet_quaternions = self.link_quaternions[:, self._robot.foot_links_idx].reshape(-1, 4)
        self.feet_orientation[:] = quat_apply(
            quat_inv(feet_quaternions), self.global_gravity.repeat(2, 1)
        ).reshape(self.num_envs, len(self._robot.foot_links_idx), 3)

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
                    "dof_pos_limits": self.dof_pos_limits,
                    "commands": self.commands,
                    "feet_first_contact": self.feet_first_contact,
                    "feet_air_time": self.feet_air_time,
                    "feet_height": self.feet_height,
                    "feet_z_velocity": self.feet_z_velocity,
                    "feet_contact_force": self.feet_contact_force,
                    "feet_orientation": self.feet_orientation,
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

    def _resample_commands(self, envs_idx: torch.Tensor) -> None:
        self.commands[envs_idx, :] = torch.rand(len(envs_idx), 3, device=self._device)
        self.commands[:, 1] *= 0
        self.commands[:, :2] *= torch.norm(self.commands[:, :2], dim=-1, keepdim=True) > 0.3
        self.commands[:, 2] *= torch.abs(self.commands[:, 2]) > 0.3
