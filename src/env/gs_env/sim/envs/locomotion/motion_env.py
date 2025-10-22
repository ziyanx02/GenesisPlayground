import os
import pickle

import torch
import yaml
from tqdm import tqdm

#
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_inv,
    quat_to_angle_axis,
    quat_to_euler,
    quat_to_rotation_6D,
    slerp,
)
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class MotionLib: ...


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
    ) -> None:
        # Initialize base legged-robot environment
        self._args = args
        super().__init__(
            args=args,
            num_envs=num_envs,
            show_viewer=show_viewer,
            device=device,
            eval_mode=eval_mode,
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
        self.link_positions_local = torch.zeros(
            self.num_envs, self._robot.n_links, 3, device=self._device
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
        self.ref_body_pos = torch.zeros(self.num_envs, self._robot.n_links, 3, device=self._device)
        self.ref_body_pos_local = torch.zeros(
            self.num_envs, self._robot.n_links, 3, device=self._device
        )

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # Motion library and reference buffers
        self._motion_lib = MotionLib(motion_file=self._args.motion_file, device=self._device)

        # per-env motion selection and time offset
        self._motion_ids = torch.zeros(self.num_envs, device=self._device, dtype=torch.long)
        self._motion_time_offsets = torch.zeros(
            self.num_envs, device=self._device, dtype=torch.float32
        )

        # initialize once
        self.reset_idx(torch.arange(self.num_envs, device=self._device, dtype=torch.long))

    def _reset_buffers(self, envs_idx: torch.IntTensor) -> None:
        super()._reset_buffers(envs_idx=envs_idx)
        self.feet_air_time[envs_idx] = 0.0

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        # set reference motion first
        self._reset_ref_motion(envs_idx=envs_idx)
        self.hard_sync_motion(envs_idx=envs_idx)

        self._reset_buffers(envs_idx=envs_idx)

    def get_terminated(self) -> torch.Tensor:
        reset_buf = self.get_truncated()
        tilt_mask = torch.logical_or(
            torch.abs(self.base_euler[:, 0]) > 0.5,
            torch.abs(self.base_euler[:, 1]) > 1.0,
        )
        height_mask = self.base_pos[:, 2] < 0.5
        reset_buf |= tilt_mask
        reset_buf |= height_mask
        contact_force_mask = torch.any(
            torch.norm(self.link_contact_forces[:, self._terminate_link_idx_local, :], dim=-1)
            > 1.0,
            dim=-1,
        )
        reset_buf |= contact_force_mask
        self.reset_buf[:] = reset_buf
        termination_dict = {}
        termination_dict["tilt"] = tilt_mask.clone()
        termination_dict["base_height"] = height_mask.clone()
        termination_dict["contact_force"] = contact_force_mask.clone()
        termination_dict["any"] = reset_buf.clone()
        self._extra_info["termination"] = termination_dict
        return reset_buf

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action=action)
        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.feet_contact
        self.feet_air_time += self.dt

    def _pre_step(self) -> None:
        super()._pre_step()

    def _update_buffers(self) -> None:
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
        self.link_positions_local[:] = self.batched_global_to_local(
            self.base_pos, self.base_quat, self.link_positions
        )
        self.link_quaternions[:] = self._robot.link_quaternions
        self.link_velocities[:] = self._robot.link_velocities

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
        motion_times = self._motion_lib.sample_motion_times(motion_ids)
        self._motion_ids[envs_idx] = motion_ids
        self._motion_time_offsets[envs_idx] = motion_times

        self._update_ref_motion(envs_idx=envs_idx, motion_ids=motion_ids, motion_times=motion_times)

        # lift slightly to avoid penetration
        # base_pos[:, 2] += 0.05

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
        base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel, body_pos_local = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )
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
        self.ref_dof_pos[envs_idx] = dof_pos
        self.ref_dof_vel[envs_idx] = dof_vel
        # self.ref_body_pos[envs_idx] = body_pos_local
        # self.ref_body_pos_local[envs_idx] = self.batched_global_to_local(self.ref_base_pos[envs_idx], self.ref_base_quat[envs_idx], body_pos_local)
        _ = body_pos_local

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


class MotionLib:
    # CREDITS: https://github.com/YanjieZe/TWIST
    def __init__(self, motion_file: str, device: torch.device) -> None:
        self._device = device

        gsplayground_order = [
            # Left Lower body 0:6
            "left_hip_roll_joint",
            "left_hip_pitch_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_roll_joint",
            "left_ankle_pitch_joint",
            # Right Lower body 6:12
            "right_hip_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_roll_joint",
            "right_ankle_pitch_joint",
            # Waist 12:15
            "waist_roll_joint",
            "waist_pitch_joint",
            "waist_yaw_joint",
            # Left Upper body 12:19
            "left_shoulder_roll_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            # Right Upper body 19:26
            "right_shoulder_roll_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        twist_order = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
        ]
        self._dof_index = []
        for dof_name in twist_order:
            self._dof_index.append(gsplayground_order.index(dof_name))

        self._load_motions(motion_file)

    def _load_motions(self, motion_file: str) -> None:
        self._motion_names = []
        self._motion_files = []
        self._body_link_list = []

        motion_weights = []
        motion_fps = []
        motion_dt = []
        motion_num_frames = []
        motion_lengths = []

        motion_base_pos = []
        motion_base_pos_delta = []
        motion_base_quat = []
        motion_base_lin_vel = []
        motion_base_ang_vel = []
        motion_dof_pos = []
        motion_dof_vel = []

        motion_local_body_pos = []

        full_motion_files, full_motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(full_motion_files)

        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = full_motion_files[i]
            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)

                    if len(self._body_link_list) == 0:
                        self._body_link_list = motion_data["link_body_list"]

                    base_pos = torch.tensor(
                        motion_data["root_pos"], dtype=torch.float, device=self._device
                    )  # (num_frames, 3)
                    base_pos_delta = base_pos[-1] - base_pos[0]
                    base_pos_delta[..., -1] = 0.0  # TODO: why?
                    base_quat = torch.tensor(
                        motion_data["root_rot"], dtype=torch.float, device=self._device
                    )  # TODO: convert to wxyz

                    fps = motion_data["fps"]
                    dt = 1.0 / fps
                    num_frames = base_pos.shape[0]
                    length = dt * (num_frames - 1)

                    base_lin_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
                    base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
                    base_lin_vel[-1, :] = base_lin_vel[-2, :]
                    base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

                    base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
                    base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
                    base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
                    base_ang_vel[-1, :] = base_ang_vel[-2, :]
                    base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

                    dof_pos = torch.tensor(
                        motion_data["dof_pos"], dtype=torch.float, device=self._device
                    )
                    dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
                    dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                    dof_vel[-1, :] = dof_vel[-2, :]
                    dof_vel = self.smooth(dof_vel, 19, device=self._device)

                    local_body_pos = torch.tensor(
                        motion_data["local_body_pos"], dtype=torch.float, device=self._device
                    )

                    self._motion_names.append(os.path.basename(curr_file))
                    self._motion_files.append(curr_file)

                    motion_weights.append(full_motion_weights[i])
                    motion_fps.append(fps)
                    motion_dt.append(dt)
                    motion_num_frames.append(num_frames)
                    motion_lengths.append(length)

                    motion_base_pos.append(base_pos)
                    motion_base_pos_delta.append(base_pos_delta)
                    motion_base_quat.append(base_quat)
                    motion_base_lin_vel.append(base_lin_vel)
                    motion_base_ang_vel.append(base_ang_vel)
                    motion_dof_pos.append(dof_pos)
                    motion_dof_vel.append(dof_vel)

                    motion_local_body_pos.append(local_body_pos)
            except Exception as e:
                print("Error loading motion file %s: %s", curr_file, e)
                continue

        assert len(self._body_link_list) > 0, "Body link list is empty"

        motion_weights = torch.tensor(motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights = motion_weights / torch.sum(motion_weights)
        self._motion_fps = torch.tensor(motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self._device
        )
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float, device=self._device)

        self._motion_base_pos = torch.cat(motion_base_pos, dim=0)
        self._motion_base_pos_delta = torch.stack(motion_base_pos_delta, dim=0)
        self._motion_base_quat = torch.cat(motion_base_quat, dim=0)
        motion_base_quat = torch.cat(motion_base_quat, dim=0)
        self._motion_base_quat = torch.cat(
            [motion_base_quat[:, 3:], motion_base_quat[:, :3]], dim=1
        )
        self._motion_base_lin_vel = torch.cat(motion_base_lin_vel, dim=0)
        self._motion_base_ang_vel = torch.cat(motion_base_ang_vel, dim=0)
        self._motion_dof_pos = torch.zeros(
            torch.cat(motion_dof_pos, dim=0).shape[0], 29, device=self._device
        )
        self._motion_dof_pos[:, self._dof_index] = torch.cat(motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.zeros(
            torch.cat(motion_dof_vel, dim=0).shape[0], 29, device=self._device
        )
        self._motion_dof_vel[:, self._dof_index] = torch.cat(motion_dof_vel, dim=0)

        self._motion_local_body_pos = torch.cat(motion_local_body_pos, dim=0)

        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)  # prefix sum of num frames

        self._motion_ids = torch.arange(self.num_motions, dtype=torch.long, device=self._device)

        print(
            f"Loaded {self.num_motions:d} motions with a total length of {self.total_length:.3f}s."
        )

    def sample_motion_ids(
        self, n: int, motion_difficulty: torch.Tensor | None = None
    ) -> torch.Tensor:
        if motion_difficulty is not None:
            motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights
        motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
        return motion_ids

    def sample_motion_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]

        motion_times = motion_len * phase
        return motion_times

    def _fetch_motion_files(self, motion_file: str) -> tuple[list[str], list[float]]:
        if motion_file.endswith(".yaml"):
            motion_files = []
            motion_weights = []
            with open(motion_file) as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_base_path = motion_config["root_path"]
            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = os.path.join(motion_base_path, motion_entry["file"])
                curr_weight = motion_entry["weight"]
                assert curr_weight >= 0

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]

        return motion_files, motion_weights

    def _calc_frame_blend(
        self, motion_ids: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self._motion_num_frames[motion_ids]

        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx

        return frame_idx0, frame_idx1, blend

    def calc_motion_frame(
        self, motion_ids: torch.Tensor, motion_times: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

        base_pos0 = self._motion_base_pos[frame_idx0]
        base_pos1 = self._motion_base_pos[frame_idx1]

        base_quat0 = self._motion_base_quat[frame_idx0]
        base_quat1 = self._motion_base_quat[frame_idx1]

        base_lin_vel = self._motion_base_lin_vel[frame_idx0]
        base_ang_vel = self._motion_base_ang_vel[frame_idx0]

        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]

        local_key_body_pos0 = self._motion_local_body_pos[frame_idx0]
        local_key_body_pos1 = self._motion_local_body_pos[frame_idx1]

        dof_vel = self._motion_dof_vel[frame_idx0]

        blend_unsqueeze = blend.unsqueeze(-1)
        base_pos = (1.0 - blend_unsqueeze) * base_pos0 + blend_unsqueeze * base_pos1
        base_pos += motion_loop_num.unsqueeze(-1) * self._motion_base_pos_delta[motion_ids]
        base_quat = slerp(base_quat0, base_quat1, blend)

        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1

        local_key_body_pos = (
            1.0 - blend_unsqueeze.unsqueeze(1)
        ) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1

        return base_pos, base_quat, base_lin_vel, base_ang_vel, dof_pos, dof_vel, local_key_body_pos

    def get_key_body_idx(self, key_body_names: list[str]) -> list[int]:
        key_body_idx = []
        for key_body_name in key_body_names:
            key_body_idx.append(self._body_link_list.index(key_body_name))
        return key_body_idx  # list

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_num_frames[motion_ids]

    def get_motion_fps(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_fps[motion_ids]

    def get_motion_dt(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_dt[motion_ids]

    def get_motion_weights(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_weights[motion_ids]

    @staticmethod
    def smooth(x: torch.Tensor, box_pts: int, device: torch.device) -> torch.Tensor:
        box = torch.ones(box_pts, device=device) / box_pts
        num_channels = x.shape[1]
        x_reshaped = x.T.unsqueeze(0)
        smoothed = torch.nn.functional.conv1d(
            x_reshaped,
            box.view(1, 1, -1).expand(num_channels, 1, -1),
            groups=num_channels,
            padding="same",
        )
        return smoothed.squeeze(0).T

    @property
    def num_motions(self) -> int:
        return self._motion_weights.shape[0]

    @property
    def motion_names(self) -> list[str]:
        return self._motion_names

    @property
    def total_length(self) -> float:
        return torch.sum(self._motion_lengths).item()
