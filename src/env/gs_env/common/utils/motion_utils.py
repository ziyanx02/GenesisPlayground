import os
import pickle

import torch
import yaml
from tqdm import tqdm

#
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_diff,
    quat_from_euler,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
    slerp,
)

_DEFAULT_DEVICE = torch.device("cpu")


# class MotionLib:
#     # CREDITS: https://github.com/YanjieZe/TWIST
#     def __init__(
#         self, motion_file: str | None = None, device: torch.device = _DEFAULT_DEVICE
#     ) -> None:
#         self._device = device
#         if motion_file is not None:
#             self._load_motions(motion_file)

#     def _load_motions(self, motion_file: str) -> None:
#         self._motion_names = []
#         self._motion_files = []
#         self._link_names = []
#         self._dof_names = []

#         motion_weights = []
#         motion_fps = []
#         motion_dt = []
#         motion_num_frames = []
#         motion_lengths = []

#         motion_base_pos = []
#         motion_base_quat = []
#         motion_base_lin_vel = []
#         motion_base_ang_vel = []
#         motion_dof_pos = []
#         motion_dof_vel = []
#         motion_link_pos_global = []
#         motion_link_quat_global = []
#         motion_link_pos_local = []
#         motion_link_quat_local = []

#         full_motion_files, full_motion_weights = self._fetch_motion_files(motion_file)
#         num_motion_files = len(full_motion_files)

#         for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
#             curr_file = full_motion_files[i]
#             try:
#                 with open(curr_file, "rb") as f:
#                     motion_data = pickle.load(f)

#                     if len(self._link_names) == 0:
#                         self._link_names = motion_data["link_names"]
#                         self._dof_names = motion_data["dof_names"]

#                     base_pos = torch.tensor(
#                         motion_data["pos"], dtype=torch.float, device=self._device
#                     )
#                     base_quat = torch.tensor(
#                         motion_data["quat"], dtype=torch.float, device=self._device
#                     )

#                     fps = motion_data["fps"]
#                     dt = 1.0 / fps
#                     num_frames = base_pos.shape[0]
#                     length = dt * (num_frames - 1)

#                     base_lin_vel = torch.zeros_like(base_pos)
#                     base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
#                     base_lin_vel[-1, :] = base_lin_vel[-2, :]
#                     base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

#                     base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
#                     base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
#                     base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
#                     base_ang_vel[-1, :] = base_ang_vel[-2, :]
#                     base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

#                     dof_pos = torch.tensor(
#                         motion_data["dof_pos"], dtype=torch.float, device=self._device
#                     )
#                     dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
#                     dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
#                     dof_vel[-1, :] = dof_vel[-2, :]
#                     dof_vel = self.smooth(dof_vel, 19, device=self._device)

#                     link_pos_global = torch.tensor(
#                         motion_data["link_pos"], dtype=torch.float, device=self._device
#                     )
#                     link_quat_global = torch.tensor(
#                         motion_data["link_quat"], dtype=torch.float, device=self._device
#                     )

#                     relative_link_pos_global = link_pos_global.clone()
#                     relative_link_pos_global[:, :, :2] -= base_pos[:, None, :2]
#                     base_euler = quat_to_euler(base_quat)
#                     base_euler[:, :2] = 0.0
#                     batched_inv_quat_yaw = quat_from_euler(
#                         -base_euler[:, None, :].repeat(1, link_pos_global.shape[1], 1)
#                     )
#                     link_pos_local = quat_apply(batched_inv_quat_yaw, relative_link_pos_global)
#                     link_quat_local = quat_mul(batched_inv_quat_yaw, link_quat_global)

#                     self._motion_names.append(os.path.basename(curr_file))
#                     self._motion_files.append(curr_file)

#                     motion_weights.append(full_motion_weights[i])
#                     motion_fps.append(fps)
#                     motion_dt.append(dt)
#                     motion_num_frames.append(num_frames)
#                     motion_lengths.append(length)

#                     motion_base_pos.append(base_pos)
#                     motion_base_quat.append(base_quat)
#                     motion_base_lin_vel.append(base_lin_vel)
#                     motion_base_ang_vel.append(base_ang_vel)
#                     motion_dof_pos.append(dof_pos)
#                     motion_dof_vel.append(dof_vel)
#                     motion_link_pos_global.append(link_pos_global)
#                     motion_link_quat_global.append(link_quat_global)
#                     motion_link_pos_local.append(link_pos_local)
#                     motion_link_quat_local.append(link_quat_local)

#             except Exception as e:
#                 print("Error loading motion file %s: %s", curr_file, e)
#                 continue

#         assert len(self._link_names) > 0, "Link names list is empty"
#         assert len(self._dof_names) > 0, "Dof names list is empty"

#         motion_weights = torch.tensor(motion_weights, dtype=torch.float, device=self._device)
#         self._motion_weights = motion_weights / torch.sum(motion_weights)
#         self._motion_fps = torch.tensor(motion_fps, dtype=torch.float, device=self._device)
#         self._motion_dt = torch.tensor(motion_dt, dtype=torch.float, device=self._device)
#         self._motion_num_frames = torch.tensor(
#             motion_num_frames, dtype=torch.long, device=self._device
#         )
#         self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float, device=self._device)

#         self._motion_base_pos = torch.cat(motion_base_pos, dim=0)
#         self._motion_base_quat = torch.cat(motion_base_quat, dim=0)
#         self._motion_base_lin_vel = torch.cat(motion_base_lin_vel, dim=0)
#         self._motion_base_ang_vel = torch.cat(motion_base_ang_vel, dim=0)
#         self._motion_dof_pos = torch.cat(motion_dof_pos, dim=0)
#         self._motion_dof_vel = torch.cat(motion_dof_vel, dim=0)
#         self._motion_link_pos_global = torch.cat(motion_link_pos_global, dim=0)
#         self._motion_link_quat_global = torch.cat(motion_link_quat_global, dim=0)
#         self._motion_link_pos_local = torch.cat(motion_link_pos_local, dim=0)
#         self._motion_link_quat_local = torch.cat(motion_link_quat_local, dim=0)

#         lengths_shifted = self._motion_num_frames.roll(1)
#         lengths_shifted[0] = 0
#         self._motion_start_idx = lengths_shifted.cumsum(0)  # prefix sum of num frames

#         self._motion_ids = torch.arange(self.num_motions, dtype=torch.long, device=self._device)

#         print(
#             f"Loaded {self.num_motions:d} motions with a total length of {self.total_length:.3f}s."
#         )

#     def sample_motion_ids(
#         self, n: int, motion_difficulty: torch.Tensor | None = None
#     ) -> torch.Tensor:
#         if motion_difficulty is not None:
#             motion_prob = self._motion_weights * motion_difficulty
#         else:
#             motion_prob = self._motion_weights
#         motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
#         return motion_ids

#     def sample_motion_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         phase = torch.rand(motion_ids.shape, device=self._device)
#         motion_len = self._motion_lengths[motion_ids]

#         motion_times = motion_len * phase
#         return motion_times

#     def _fetch_motion_files(self, motion_file: str) -> tuple[list[str], list[float]]:
#         if motion_file.endswith(".yaml"):
#             motion_files = []
#             motion_weights = []
#             with open(motion_file) as f:
#                 motion_config = yaml.load(f, Loader=yaml.SafeLoader)

#             motion_base_path = motion_config["root_path"]
#             motion_list = motion_config["motions"]
#             for motion_entry in motion_list:
#                 curr_file = os.path.join(motion_base_path, motion_entry["file"])
#                 curr_weight = motion_entry["weight"]
#                 assert curr_weight >= 0

#                 motion_weights.append(curr_weight)
#                 motion_files.append(curr_file)
#         else:
#             motion_files = [motion_file]
#             motion_weights = [1.0]

#         return motion_files, motion_weights

#     def _calc_frame_blend(
#         self, motion_ids: torch.Tensor, times: torch.Tensor
#     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         num_frames = self._motion_num_frames[motion_ids]

#         phase = times / self._motion_lengths[motion_ids]
#         phase = torch.clip(phase, 0.0, 1.0)

#         frame_idx0 = (phase * (num_frames - 1)).long()
#         frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
#         blend = phase * (num_frames - 1) - frame_idx0.float()

#         frame_start_idx = self._motion_start_idx[motion_ids]
#         frame_idx0 += frame_start_idx
#         frame_idx1 += frame_start_idx

#         return frame_idx0, frame_idx1, blend

#     def calc_motion_frame(
#         self, motion_ids: torch.Tensor, motion_times: torch.Tensor
#     ) -> tuple[
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#         torch.Tensor,
#     ]:
#         assert motion_times.min() >= 0.0, "motion_times must be non-negative"
#         motion_times = torch.min(motion_times, self._motion_lengths[motion_ids])

#         frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

#         base_pos0 = self._motion_base_pos[frame_idx0]
#         base_pos1 = self._motion_base_pos[frame_idx1]

#         base_quat0 = self._motion_base_quat[frame_idx0]
#         base_quat1 = self._motion_base_quat[frame_idx1]

#         base_lin_vel = self._motion_base_lin_vel[frame_idx0]
#         base_ang_vel = self._motion_base_ang_vel[frame_idx0]

#         dof_pos0 = self._motion_dof_pos[frame_idx0]
#         dof_pos1 = self._motion_dof_pos[frame_idx1]

#         link_pos_local0 = self._motion_link_pos_local[frame_idx0]
#         link_pos_local1 = self._motion_link_pos_local[frame_idx1]

#         link_quat_local0 = self._motion_link_quat_local[frame_idx0]
#         link_quat_local1 = self._motion_link_quat_local[frame_idx1]

#         dof_vel = self._motion_dof_vel[frame_idx0]

#         blend_unsqueeze = blend.unsqueeze(-1)
#         base_pos = (1.0 - blend_unsqueeze) * base_pos0 + blend_unsqueeze * base_pos1
#         base_quat = slerp(base_quat0, base_quat1, blend)

#         dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1

#         link_pos_local = (
#             1.0 - blend_unsqueeze.unsqueeze(1)
#         ) * link_pos_local0 + blend_unsqueeze.unsqueeze(1) * link_pos_local1

#         link_quat_local = slerp(
#             link_quat_local0, link_quat_local1, blend[:, None].repeat(1, link_quat_local0.shape[1])
#         )

#         return (
#             base_pos,
#             base_quat,
#             base_lin_vel,
#             base_ang_vel,
#             dof_pos,
#             dof_vel,
#             link_pos_local,
#             link_quat_local,
#         )

#     def get_link_idx_local_by_name(self, name: str) -> int:
#         return self._link_names.index(name)

#     def get_joint_idx_by_name(self, name: str) -> int:
#         return self._dof_names.index(name)

#     def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         return self._motion_lengths[motion_ids]

#     def get_motion_num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         return self._motion_num_frames[motion_ids]

#     def get_motion_fps(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         return self._motion_fps[motion_ids]

#     def get_motion_dt(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         return self._motion_dt[motion_ids]

#     def get_motion_weights(self, motion_ids: torch.Tensor) -> torch.Tensor:
#         return self._motion_weights[motion_ids]

#     @staticmethod
#     def smooth(x: torch.Tensor, box_pts: int, device: torch.device) -> torch.Tensor:
#         box = torch.ones(box_pts, device=device) / box_pts
#         num_channels = x.shape[1]
#         x_reshaped = x.T.unsqueeze(0)
#         smoothed = torch.nn.functional.conv1d(
#             x_reshaped,
#             box.view(1, 1, -1).expand(num_channels, 1, -1),
#             groups=num_channels,
#             padding="same",
#         )
#         return smoothed.squeeze(0).T

#     @property
#     def num_motions(self) -> int:
#         return self._motion_weights.shape[0]

#     @property
#     def motion_names(self) -> list[str]:
#         return self._motion_names

#     @property
#     def total_length(self) -> float:
#         return torch.sum(self._motion_lengths).item()


class MotionLib:
    def __init__(
        self,
        motion_file: str | None = None,
        device: torch.device = _DEFAULT_DEVICE,
        target_fps: float = 50.0,
    ) -> None:
        self._device = device
        self._target_fps = target_fps
        self._motion_obs_steps = None
        if motion_file is not None:
            self._load_motions(motion_file)

    def _load_motions(self, motion_file: str) -> None:
        self._motion_names = []
        self._motion_files = []
        self._link_names = []
        self._dof_names = []

        motion_weights = []
        motion_num_frames = []
        motion_lengths = []

        motion_base_pos = []
        motion_base_quat = []
        motion_base_lin_vel = []
        motion_base_ang_vel = []
        motion_dof_pos = []
        motion_dof_vel = []
        motion_link_pos_global = []
        motion_link_quat_global = []
        motion_link_pos_local = []
        motion_link_quat_local = []
        motion_foot_contact = []

        full_motion_files, full_motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(full_motion_files)

        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = full_motion_files[i]
            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)

                    if len(self._link_names) == 0:
                        self._link_names = motion_data["link_names"]
                        self._dof_names = motion_data["dof_names"]
                        self._foot_link_indices = motion_data["foot_link_indices"]

                    base_pos = torch.tensor(
                        motion_data["pos"], dtype=torch.float, device=self._device
                    )
                    base_quat = torch.tensor(
                        motion_data["quat"], dtype=torch.float, device=self._device
                    )

                    fps = motion_data["fps"]
                    dt = 1.0 / fps
                    num_frames = base_pos.shape[0]
                    length = dt * (num_frames - 1)

                    base_lin_vel = torch.zeros_like(base_pos)
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

                    link_pos_global = torch.tensor(
                        motion_data["link_pos"], dtype=torch.float, device=self._device
                    )
                    link_quat_global = torch.tensor(
                        motion_data["link_quat"], dtype=torch.float, device=self._device
                    )

                    foot_contact = torch.tensor(
                        motion_data["foot_contact"], dtype=torch.float, device=self._device
                    )

                    # Resample to target FPS if requested
                    target_fps_curr = float(self._target_fps)

                    if abs(target_fps_curr - fps) > 1e-6:
                        # time length stays the same
                        new_num_frames = int(round(length * target_fps_curr)) + 1
                        t = torch.linspace(0.0, length, steps=new_num_frames, device=self._device)
                        # compute blend weights against original frames
                        phase = torch.clip(t / length, 0.0, 1.0)
                        idx0 = (phase * (num_frames - 1)).long()
                        idx1 = torch.min(
                            idx0 + 1, torch.tensor(num_frames - 1, device=self._device)
                        )
                        blend = phase * (num_frames - 1) - idx0.float()
                        blend_u = blend.unsqueeze(-1)

                        # positions, dof: linear
                        base_pos = (1.0 - blend_u) * base_pos[idx0] + blend_u * base_pos[idx1]
                        dof_pos = (1.0 - blend_u) * dof_pos[idx0] + blend_u * dof_pos[idx1]
                        foot_contact = 1 - (1 - foot_contact[idx0]) * (1 - foot_contact[idx1])
                        link_pos_global = (1.0 - blend_u.unsqueeze(1)) * link_pos_global[
                            idx0
                        ] + blend_u.unsqueeze(1) * link_pos_global[idx1]

                        # quaternions: slerp
                        base_quat = slerp(base_quat[idx0], base_quat[idx1], blend)
                        link_quat_global = slerp(
                            link_quat_global[idx0],
                            link_quat_global[idx1],
                            blend[:, None].repeat(1, link_quat_global.shape[1]),
                        )

                        # update meta based on resampled length
                        fps = target_fps_curr
                        dt = 1.0 / fps
                        num_frames = base_pos.shape[0]
                        length = dt * (num_frames - 1)
                    else:
                        # ensure library fps is set
                        fps = target_fps_curr
                        dt = 1.0 / fps

                    # recompute velocities at current fps
                    base_lin_vel = torch.zeros_like(base_pos)
                    base_lin_vel[:-1, :] = fps * (base_pos[1:, :] - base_pos[:-1, :])
                    base_lin_vel[-1, :] = base_lin_vel[-2, :]
                    base_lin_vel = self.smooth(base_lin_vel, 19, device=self._device)

                    base_ang_vel = torch.zeros_like(base_pos)  # (num_frames, 3)
                    base_dquat = quat_diff(base_quat[:-1], base_quat[1:])
                    base_ang_vel[:-1, :] = fps * quat_to_angle_axis(base_dquat)
                    base_ang_vel[-1, :] = base_ang_vel[-2, :]
                    base_ang_vel = self.smooth(base_ang_vel, 19, device=self._device)

                    dof_vel = torch.zeros_like(dof_pos)  # (num_frames, num_dof)
                    dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                    dof_vel[-1, :] = dof_vel[-2, :]
                    dof_vel = self.smooth(dof_vel, 19, device=self._device)

                    # recompute local link transforms with yaw-only removal from base
                    relative_link_pos_global = link_pos_global.clone()
                    relative_link_pos_global[:, :, :2] -= base_pos[:, None, :2]
                    base_euler = quat_to_euler(base_quat)
                    base_euler[:, :2] = 0.0
                    batched_inv_quat_yaw = quat_from_euler(
                        -base_euler[:, None, :].repeat(1, link_pos_global.shape[1], 1)
                    )
                    link_pos_local = quat_apply(batched_inv_quat_yaw, relative_link_pos_global)
                    link_quat_local = quat_mul(batched_inv_quat_yaw, link_quat_global)

                    self._motion_names.append(os.path.basename(curr_file))
                    self._motion_files.append(curr_file)

                    motion_weights.append(full_motion_weights[i])
                    motion_num_frames.append(num_frames)
                    motion_lengths.append(length)

                    motion_base_pos.append(base_pos)
                    motion_base_quat.append(base_quat)
                    motion_base_lin_vel.append(base_lin_vel)
                    motion_base_ang_vel.append(base_ang_vel)
                    motion_dof_pos.append(dof_pos)
                    motion_dof_vel.append(dof_vel)
                    motion_link_pos_global.append(link_pos_global)
                    motion_link_quat_global.append(link_quat_global)
                    motion_link_pos_local.append(link_pos_local)
                    motion_link_quat_local.append(link_quat_local)
                    motion_foot_contact.append(foot_contact)

            except Exception as e:
                print(f"Error loading motion file {curr_file}: {e}")
                continue

        assert len(self._link_names) > 0, "Link names list is empty"
        assert len(self._dof_names) > 0, "Dof names list is empty"

        motion_weights = torch.tensor(motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights = motion_weights / torch.sum(motion_weights)
        self._motion_num_frames = torch.tensor(
            motion_num_frames, dtype=torch.long, device=self._device
        )
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float, device=self._device)

        self._motion_base_pos = torch.cat(motion_base_pos, dim=0)
        self._motion_base_quat = torch.cat(motion_base_quat, dim=0)
        self._motion_base_lin_vel = torch.cat(motion_base_lin_vel, dim=0)
        self._motion_base_ang_vel = torch.cat(motion_base_ang_vel, dim=0)
        self._motion_dof_pos = torch.cat(motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(motion_dof_vel, dim=0)
        self._motion_link_pos_global = torch.cat(motion_link_pos_global, dim=0)
        self._motion_link_quat_global = torch.cat(motion_link_quat_global, dim=0)
        self._motion_link_pos_local = torch.cat(motion_link_pos_local, dim=0)
        self._motion_link_quat_local = torch.cat(motion_link_quat_local, dim=0)
        self._motion_foot_contact = torch.cat(motion_foot_contact, dim=0)

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
        # Sample integer steps uniformly and convert to times by dividing by fps
        n_steps = self._motion_num_frames[motion_ids] - 1
        phase = torch.rand(motion_ids.shape, device=self._device)
        steps = torch.round(phase * n_steps.float()).long()
        steps = torch.clamp(steps, min=0)  # safety
        motion_times = steps.float() / float(self.fps)
        return motion_times

    def _fetch_motion_files(
        self, motion_file: str, motion_weight: float = 1.0
    ) -> tuple[list[str], list[float]]:
        # Recursively expand YAML motion manifests into flat file and weight lists.
        if motion_file.endswith(".yaml"):
            all_files: list[str] = []
            all_weights: list[float] = []
            with open(motion_file) as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            motion_base_path = motion_config["root_path"]
            motion_list = motion_config["motions"]
            for motion_entry in motion_list:
                curr_file = os.path.join(motion_base_path, motion_entry["file"])
                curr_weight = float(motion_entry.get("weight", 1.0))
                assert curr_weight >= 0
                sub_files, sub_weights = self._fetch_motion_files(
                    curr_file, curr_weight * motion_weight
                )
                all_files.extend(sub_files)
                all_weights.extend(sub_weights)
            return all_files, all_weights
        else:
            return [motion_file], [motion_weight]

    def set_observed_steps(self, observed_steps: dict[str, list[int]]) -> None:
        obs_terms = {
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
            "dof_pos",
            "dof_vel",
            "link_pos_local",
            "link_quat_local",
            "foot_contact",
        }
        self._motion_obs_steps = {}
        for term in obs_terms:
            if term in observed_steps.keys():
                self._motion_obs_steps[term] = torch.tensor(
                    observed_steps[term], dtype=torch.long, device=self._device
                )
            else:
                self._motion_obs_steps[term] = torch.tensor(
                    [
                        1,
                    ],
                    dtype=torch.long,
                    device=self._device,
                )

    def get_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]

        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
        )

    def get_ref_motion_frame(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        assert motion_times.min() >= 0.0, "motion_times must be non-negative"
        # snap to discrete frame grid using unified fps and clamp within motion length
        fps = self.fps
        motion_len = self._motion_lengths[motion_ids] - 1.0 / fps
        motion_times = torch.min(motion_times, motion_len)
        steps = torch.round(motion_times * fps).long()

        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx = frame_start_idx + steps + 1

        base_pos = self._motion_base_pos[frame_idx]
        base_quat = self._motion_base_quat[frame_idx]
        base_lin_vel = self._motion_base_lin_vel[frame_idx]
        base_ang_vel = self._motion_base_ang_vel[frame_idx]
        dof_pos = self._motion_dof_pos[frame_idx]
        dof_vel = self._motion_dof_vel[frame_idx]
        link_pos_local = self._motion_link_pos_local[frame_idx]
        link_quat_local = self._motion_link_quat_local[frame_idx]
        foot_contact = self._motion_foot_contact[frame_idx]

        # Assemble future-steps observations if configured
        obs_dict: dict[str, torch.Tensor] = {}
        if self._motion_obs_steps is not None:
            max_steps = self._motion_num_frames[motion_ids] - 1

            steps_map = self._motion_obs_steps

            def get_obs_tensor(term: str) -> torch.Tensor:
                steps_tensor = steps_map[term]
                future_steps = steps[:, None] + steps_tensor[None, :]
                future_steps = torch.minimum(future_steps, max_steps[:, None])
                future_idx_local = frame_start_idx[:, None] + future_steps  # (B, K)
                B, K = future_idx_local.shape
                tensor = getattr(self, f"_motion_{term}")
                extra_shape = tensor.shape[1:]
                flat = tensor[future_idx_local.reshape(-1)]
                return flat.reshape(B, K, *extra_shape)

            for key in self._motion_obs_steps.keys():
                obs_dict[key] = get_obs_tensor(key)

        return (
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            dof_pos,
            dof_vel,
            link_pos_local,
            link_quat_local,
            foot_contact,
            obs_dict,
        )

    def get_link_idx_local_by_name(self, name: str) -> int:
        return self._link_names.index(name)

    def get_joint_idx_by_name(self, name: str) -> int:
        return self._dof_names.index(name)

    def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_lengths[motion_ids]

    def get_motion_num_frames(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self._motion_num_frames[motion_ids]

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
    def link_names(self) -> list[str]:
        return self._link_names

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def foot_link_indices(self) -> list[int]:
        return self._foot_link_indices

    @property
    def num_motions(self) -> int:
        return self._motion_weights.shape[0]

    @property
    def motion_names(self) -> list[str]:
        return self._motion_names

    @property
    def total_length(self) -> float:
        return torch.sum(self._motion_lengths).item()

    @property
    def fps(self) -> float:
        return self._target_fps
