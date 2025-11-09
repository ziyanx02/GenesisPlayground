import torch

from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_euler,
    quat_to_rotation_6D,
)

#
from gs_env.common.utils.motion_utils import MotionLib
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class MotionEnv(LeggedRobotEnv):
    """
    Motion imitation environment using reference motions.

    Exposes the following additional tensors so they can be used in
    observations and rewards via `actor_obs_terms` / `critic_obs_terms`:
      - ref_base_pos: (B, 3)
      - ref_base_quat: (B, 4)
      - ref_base_lin_vel: (B, 3)
      - ref_base_ang_vel: (B, 3)
      - ref_dof_pos: (B, D)
      - ref_dof_vel: (B, D)
      - ref_body_pos: (B, L, 3) global positions of key bodies in the ref
      - mimic_obs: (B, M) compact imitation observation (current frame)
    """

    def __init__(
        self,
        args: MotionEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
        debug: bool = False,
    ) -> None:
        # Initialize base legged-robot environment
        self._args = args
        super().__init__(
            args=args,
            num_envs=num_envs,
            show_viewer=show_viewer,
            device=device,
            eval_mode=eval_mode,
            debug=debug,
        )

    def _init(self) -> None:
        # Pre-parent: allocate feet-related buffers optionally used by rewards
        self.feet_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_contact_force = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_first_contact = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.feet_air_time = torch.zeros(
            (self.num_envs, len(self._robot.foot_links_idx)),
            device=self._device,
            dtype=torch.float32,
        )
        self.base_height = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        self.base_rotation_6D = torch.zeros(
            self.num_envs, 6, device=self._device, dtype=torch.float32
        )
        self.base_lin_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.base_ang_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.link_pos_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 3, device=self._device
        )
        self.link_quat_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 4, device=self._device
        )

        # reference trajectories (current frame)
        self.ref_base_pos = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_quat = torch.zeros(self.num_envs, 4, device=self._device)
        self.ref_base_height = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)
        self.ref_base_rotation_6D = torch.zeros(
            self.num_envs, 6, device=self._device, dtype=torch.float32
        )
        self.ref_base_euler = torch.zeros(
            self.num_envs, 3, device=self._device, dtype=torch.float32
        )
        self.ref_base_lin_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_ang_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_lin_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_base_ang_vel_local = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_dof_pos = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.ref_dof_vel = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.ref_link_pos_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 3, device=self._device
        )
        self.ref_link_quat_local_yaw = torch.zeros(
            self.num_envs, self._robot.n_links, 4, device=self._device
        )

        # Motion library and reference buffers
        self._motion_lib = MotionLib(motion_file=self._args.motion_file, device=self._device)
        if self._args.motion_file is not None:
            self.ref_link_idx_local = [
                self._motion_lib.get_link_idx_local_by_name(name) for name in self.robot.link_names
            ]
            self.ref_joint_idx_local = [
                self._motion_lib.get_joint_idx_by_name(name) for name in self.robot.dof_names
            ]

        # tracking link indices
        tracking_link_names = self._args.tracking_link_names
        self.tracking_link_idx_local = (
            [self._motion_lib.get_link_idx_local_by_name(name) for name in tracking_link_names]
            if self._args.motion_file is not None
            else []
        )

        self.tracking_link_pos_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.tracking_link_quat_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )
        self.ref_tracking_link_pos_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 3, device=self._device
        )
        self.ref_tracking_link_quat_local_yaw = torch.zeros(
            self.num_envs, len(self.tracking_link_idx_local), 4, device=self._device
        )

        # per-env motion selection and time offset
        self._motion_ids = torch.zeros(self.num_envs, device=self._device, dtype=torch.long)
        self._motion_time_offsets = torch.zeros(
            self.num_envs, device=self._device, dtype=torch.float32
        )
        self._motion_lengths = torch.zeros(self.num_envs, device=self._device, dtype=torch.float32)

        # termination
        error_list = [
            "base_pos_error",
            "base_height_error",
            "base_quat_error",
            "dof_pos_error",
            "tracking_link_pos_error",
        ]
        self._terminate_after_error = {}
        self._min_terminate_after_error = {}
        self._max_terminate_after_error = {}
        self._num_step_since_update_terminate_error = 0
        self._error_mask_buffer = {}
        for error_name in error_list:
            terminate_after_error = self._args.terminate_after_error[error_name][0]
            min_terminate_after_error = self._args.terminate_after_error[error_name][1]
            max_terminate_after_error = self._args.terminate_after_error[error_name][0]
            if terminate_after_error and min_terminate_after_error:
                self._terminate_after_error[error_name] = terminate_after_error
                self._min_terminate_after_error[error_name] = min_terminate_after_error
                self._max_terminate_after_error[error_name] = max_terminate_after_error
                self._error_mask_buffer[error_name] = []

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # initialize once
        self.reset_idx(torch.IntTensor(range(self.num_envs)))

    def _reset_buffers(self, envs_idx: torch.IntTensor) -> None:
        super()._reset_buffers(envs_idx=envs_idx)
        self.feet_air_time[envs_idx] = 0.0

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        if self._eval_mode:
            self.reset_to_default_pos(envs_idx)
            return

        # set reference motion first
        self.time_since_reset[envs_idx] = 0.0
        self._reset_ref_motion(envs_idx=envs_idx)
        self.hard_sync_motion(envs_idx=envs_idx)

        num_selected = int(self._args.reset_to_default_pose_ratio * len(envs_idx))
        if num_selected > 0:
            local_idx = torch.randperm(len(envs_idx), device=envs_idx.device)[:num_selected]
            envs_selected = envs_idx[local_idx]

            # Set motion time to 0.0 by zeroing time_since_reset for selected
            self._motion_time_offsets[envs_selected] = 0.0

            # Update reference at t=0 for these envs
            self._update_ref_motion(
                envs_idx=envs_selected,
                motion_ids=self._motion_ids[envs_selected],
                motion_times=self._motion_time_offsets[envs_selected],
            )

            self.reset_to_default_pos(envs_selected)

        self._reset_buffers(envs_idx=envs_idx)

    def reset_to_default_pos(self, envs_idx: torch.Tensor) -> None:
        # different from super().reset_idx(), considering the facing direction
        default_pos = self._robot.default_pos[None, :].repeat(len(envs_idx), 1)
        default_quat = self._robot.default_quat[None, :].repeat(len(envs_idx), 1)
        default_dof_pos = self._robot.default_dof_pos[None, :].repeat(len(envs_idx), 1)

        base_pos = self.ref_base_pos[envs_idx]
        base_pos[:, 2] = default_pos[:, 2]
        random_euler = torch.zeros(len(envs_idx), 3, device=self._device)
        random_euler[:, 0] = (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_pitch_range[1] - self._args.reset_pitch_range[0])
            + self._args.reset_pitch_range[0]
        )
        random_euler[:, 1] = (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_roll_range[1] - self._args.reset_roll_range[0])
            + self._args.reset_roll_range[0]
        )
        random_euler[:, 2] = self.ref_base_euler[envs_idx, 2]
        random_euler[:, 2] += (
            torch.rand(len(envs_idx), device=self._device)
            * (self._args.reset_yaw_range[1] - self._args.reset_yaw_range[0])
            + self._args.reset_yaw_range[0]
        )
        random_quat = quat_from_euler(random_euler)
        base_quat = quat_mul(default_quat, random_quat)
        random_dof_pos = (
            torch.rand(len(envs_idx), self._robot.dof_dim, device=self._device)
            * (self._args.reset_dof_pos_range[1] - self._args.reset_dof_pos_range[0])
            + self._args.reset_dof_pos_range[0]
        )
        dof_pos = default_dof_pos + random_dof_pos

        # Apply state to robot for selected envs
        self._robot.set_state(pos=base_pos, quat=base_quat, dof_pos=dof_pos, envs_idx=envs_idx)

    def get_terminated(self) -> torch.Tensor:
        # termination dictionary for extra info
        termination_dict = {}

        reset_buf = self.get_truncated()

        # tilt_mask = torch.logical_or(
        #     torch.abs(self.base_euler[:, 0]) > 0.5,
        #     torch.abs(self.base_euler[:, 1]) > 1.0,
        # )
        # reset_buf |= tilt_mask

        # height_mask = self.base_pos[:, 2] < 0.3
        # reset_buf |= height_mask

        contact_force_mask = torch.any(
            torch.norm(self.link_contact_forces[:, self._terminate_link_idx_local, :], dim=-1)
            > 1.0,
            dim=-1,
        )
        reset_buf |= contact_force_mask

        # terminate if motino_time will exceed motion length after next step
        # avoid passing overlimit motion time to calc_motion_frame
        motion_end_mask = self.motion_times + self.dt > self._motion_lengths
        reset_buf |= motion_end_mask

        # Only enable error-based termination after a certain motion time, if specified
        terminate_by_error = self.motion_times > self._args.no_terminate_before_motion_time
        terminate_by_error |= (
            self.time_since_random_push > self._args.no_terminate_before_motion_time
        )
        base_pos_error = torch.norm(self.base_pos - self.ref_base_pos, dim=-1)
        base_height_error = torch.abs(self.base_height - self.ref_base_height)
        base_quat_error = quat_error_magnitude(self.base_quat, self.ref_base_quat)
        dof_pos_error = torch.sum(torch.abs(self.dof_pos - self.ref_dof_pos), dim=-1)
        tracking_link_pos_error = torch.norm(
            self.tracking_link_pos_local_yaw - self.ref_tracking_link_pos_local_yaw, dim=-1
        ).mean(dim=-1)

        error_dict = {}
        error_dict["base_pos_error"] = base_pos_error.clone()
        error_dict["base_height_error"] = base_height_error.clone()
        error_dict["base_quat_error"] = base_quat_error.clone()
        error_dict["dof_pos_error"] = dof_pos_error.clone()
        error_dict["tracking_link_pos_error"] = tracking_link_pos_error.clone()

        error_mask = {}
        for error_name in self._terminate_after_error.keys():
            error_mask[error_name] = (
                error_dict[error_name] > self._terminate_after_error[error_name]
            )
            termination_dict[f"{error_name}"] = error_mask[error_name].clone()
        if len(self._terminate_after_error.keys()) > 0:
            terminate_by_error &= torch.any(torch.stack(list(error_mask.values())), dim=0)
        if not self._eval_mode:
            reset_buf |= terminate_by_error

        if self._args.adaptive_termination_ratio is not None:
            self._update_terminate_error(error_mask)

        # for error_name in self._terminate_after_error.keys():
        #     if error_mask[error_name][0]:
        #         print(f"terminate by {error_name}")

        self.reset_buf[:] = reset_buf

        # termination_dict["tilt"] = tilt_mask.clone()
        # termination_dict["height"] = height_mask.clone()
        # termination_dict["motion_end"] = motion_end_mask.clone()
        # termination_dict["terminate_by_error"] = terminate_by_error.clone()
        termination_dict["contact_force"] = contact_force_mask.clone()
        # termination_dict["any"] = reset_buf.clone()
        self._extra_info["termination"] = termination_dict

        if self._args.adaptive_termination_ratio is not None:
            for key in self._terminate_after_error.keys():
                self._extra_info["info"][f"terminate_threshold_{key}"] = (
                    self._terminate_after_error[key]
                )

        for error_name in error_dict.keys():
            error_dict[error_name] = error_dict[error_name].mean().item()
        self._extra_info["info"].update(error_dict)

        return reset_buf

    def _update_terminate_error(self, error_mask: dict[str, torch.Tensor]) -> None:
        for error_name in self._terminate_after_error.keys():
            self._error_mask_buffer[error_name].append(error_mask[error_name].float())
        self._num_step_since_update_terminate_error += 1
        if self._num_step_since_update_terminate_error >= 20:
            self._num_step_since_update_terminate_error = 0
            for error_name in self._terminate_after_error.keys():
                terminate_by_error_ratio = torch.mean(
                    torch.stack(self._error_mask_buffer[error_name])
                ).item()
                if terminate_by_error_ratio > 1.5 * self._args.adaptive_termination_ratio:  # type: ignore
                    self._terminate_after_error[error_name] *= 1.5
                elif terminate_by_error_ratio < 0.5 * self._args.adaptive_termination_ratio:  # type: ignore
                    self._terminate_after_error[error_name] /= 1.5
                    self._terminate_after_error[error_name] = max(
                        self._min_terminate_after_error[error_name],
                        self._terminate_after_error[error_name],
                    )
                    self._terminate_after_error[error_name] = min(
                        self._max_terminate_after_error[error_name],
                        self._terminate_after_error[error_name],
                    )
                self._error_mask_buffer[error_name] = []

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action=action)
        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.feet_contact
        self.feet_air_time += self.dt

    def _pre_step(self) -> None:
        super()._pre_step()

    def update_buffers(self) -> None:
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        self.base_height[:] = self.base_pos[:, 2]
        self.base_rotation_6D[:] = quat_to_rotation_6D(self.base_quat)
        self.base_euler[:] = quat_to_euler(self.base_quat)
        self.projected_gravity[:] = quat_apply(quat_inv(self.base_quat), self.global_gravity)
        self.base_lin_vel[:] = self._robot.get_vel()
        self.base_ang_vel[:] = self._robot.get_ang()
        self.base_lin_vel_local[:] = self.global_to_local(self.base_lin_vel)
        self.base_ang_vel_local[:] = self.global_to_local(self.base_ang_vel)

        self.link_contact_forces[:] = self._robot.link_contact_forces
        self.link_positions[:] = self._robot.link_positions
        self.link_quaternions[:] = self._robot.link_quaternions

        link_pos_local_yaw = self._robot.link_positions
        link_pos_local_yaw[:, :, :2] -= self.base_pos[:, None, :2]
        inv_quat_yaw = quat_from_angle_axis(
            -self.base_euler[:, 2], torch.tensor([0, 0, 1], device=self._device, dtype=torch.float)
        )[:, None, :].repeat(1, self._robot.n_links, 1)
        link_pos_local_yaw = quat_apply(inv_quat_yaw, link_pos_local_yaw)
        self.link_pos_local_yaw[:] = link_pos_local_yaw
        link_quat_local_yaw = quat_mul(inv_quat_yaw, self._robot.link_quaternions)
        self.link_quat_local_yaw[:] = link_quat_local_yaw
        self.link_velocities[:] = self._robot.link_velocities
        self.tracking_link_pos_local_yaw[:] = self.link_pos_local_yaw[
            :, self.tracking_link_idx_local
        ]
        self.tracking_link_quat_local_yaw[:] = self.link_quat_local_yaw[
            :, self.tracking_link_idx_local
        ]

        # contacts
        self.feet_contact_force[:] = self.link_contact_forces[:, self._robot.foot_links_idx, 2]
        self.feet_contact[:] = self.feet_contact_force > 1.0

    def update_history(self) -> None:
        super().update_history()
        self.feet_air_time *= 1 - self.feet_contact
        # update reference motion after calculating rewards
        self._update_ref_motion()

    # ---------- Motion utilities ----------
    def _reset_ref_motion(self, envs_idx: torch.Tensor) -> None:
        assert self._motion_lib is not None
        n = len(envs_idx)
        motion_ids = self._motion_lib.sample_motion_ids(n)
        motion_times = (
            self._motion_lib.sample_motion_times(motion_ids)
            * self._args.reset_to_motion_range_ratio
        )
        self._motion_ids[envs_idx] = motion_ids
        self._motion_time_offsets[envs_idx] = motion_times
        self._motion_lengths[envs_idx] = self._motion_lib.get_motion_length(motion_ids)
        self._update_ref_motion(envs_idx=envs_idx, motion_ids=motion_ids, motion_times=motion_times)

    def _update_ref_motion(
        self,
        envs_idx: torch.Tensor | None = None,
        motion_ids: torch.Tensor | None = None,
        motion_times: torch.Tensor | None = None,
    ) -> None:
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs, device=self._device, dtype=torch.long)
        if motion_ids is None:
            motion_ids = self.motion_ids
        if motion_times is None:
            motion_times = self.motion_times
        (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
            link_pos_local,
            link_quat_local,
        ) = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        self.ref_base_pos[envs_idx] = base_pos
        self.ref_base_quat[envs_idx] = base_quat
        self.ref_base_height[envs_idx] = self.ref_base_pos[envs_idx, 2]
        self.ref_base_rotation_6D[envs_idx] = quat_to_rotation_6D(base_quat)
        self.ref_base_euler[envs_idx] = quat_to_euler(base_quat)
        self.ref_base_lin_vel[envs_idx] = base_lin_vel
        self.ref_base_ang_vel[envs_idx] = base_ang_vel
        self.ref_base_lin_vel_local[envs_idx] = self.batched_global_to_local(
            self.ref_base_pos[envs_idx], self.ref_base_quat[envs_idx], base_lin_vel
        )
        self.ref_base_ang_vel_local[envs_idx] = self.batched_global_to_local(
            self.ref_base_pos[envs_idx], self.ref_base_quat[envs_idx], base_ang_vel
        )
        self.ref_dof_pos[envs_idx] = dof_pos[:, self.ref_joint_idx_local]
        self.ref_dof_vel[envs_idx] = dof_vel[:, self.ref_joint_idx_local]
        self.ref_link_pos_local_yaw[envs_idx] = link_pos_local[:, self.ref_link_idx_local]
        self.ref_link_quat_local_yaw[envs_idx] = link_quat_local[:, self.ref_link_idx_local]
        self.ref_tracking_link_pos_local_yaw[envs_idx] = self.ref_link_pos_local_yaw[envs_idx][
            :, self.tracking_link_idx_local
        ]
        self.ref_tracking_link_quat_local_yaw[envs_idx] = self.ref_link_quat_local_yaw[envs_idx][
            :, self.tracking_link_idx_local
        ]

    def hard_sync_motion(self, envs_idx: torch.Tensor) -> None:
        self._update_ref_motion()
        self._robot.set_state(
            pos=self.ref_base_pos[envs_idx],
            quat=self.ref_base_quat[envs_idx],
            dof_pos=self.ref_dof_pos[envs_idx],
            envs_idx=envs_idx,
            lin_vel=self.ref_base_lin_vel[envs_idx],
            ang_vel=self.ref_base_ang_vel[envs_idx],
            dof_vel=self.ref_dof_vel[envs_idx],
        )

    def hard_reset_motion(self, envs_idx: torch.Tensor, motion_id: int) -> None:
        if motion_id > self.motion_lib.num_motions:
            print(
                f"Motion ID {motion_id} is out of range. Valid range is 0 to {self.motion_lib.num_motions - 1}"
            )
            return
        print(f"Hard resetting motion to {self.motion_lib.motion_names[motion_id]}")
        self._motion_ids[envs_idx] = motion_id
        self._motion_time_offsets[envs_idx] = 0.0
        self._motion_lengths[envs_idx] = self._motion_lib.get_motion_length(
            self._motion_ids[envs_idx]
        )
        self.hard_sync_motion(envs_idx=envs_idx)

    @property
    def motion_lib(self) -> MotionLib:
        return self._motion_lib

    @property
    def motion_ids(self) -> torch.Tensor:
        return self._motion_ids

    @property
    def motion_times(self) -> torch.Tensor:
        return self.time_since_reset + self._motion_time_offsets
