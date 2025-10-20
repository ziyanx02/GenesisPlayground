import torch

#
from gs_env.common.utils.math_utils import (
    quat_apply,
)
from gs_env.common.utils.motion_lib import MotionLib
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

_DEFAULT_DEVICE = torch.device("cpu")


class MotionEnv(LeggedRobotEnv):
    """
    Motion imitation environment using reference motions.

    Exposes the following additional tensors so they can be used in
    observations and rewards via `actor_obs_terms` / `critic_obs_terms`:
      - ref_root_pos: (B, 3)
      - ref_root_rot: (B, 4)
      - ref_root_vel: (B, 3)
      - ref_root_ang_vel: (B, 3)
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
        # motion-related placeholders (populated in _init)
        self._motion_lib: MotionLib | None = None
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

        # Let base class set up common buffers, spaces, and rendering
        super()._init()

        # Motion library and reference buffers
        self._load_motions()
        self._init_motion_buffers()

    def _load_motions(self) -> None:
        assert hasattr(self._args, "motion_file") and self._args.motion_file, (
            "MotionEnvArgs.motion_file must be provided"
        )
        self._motion_lib = MotionLib(motion_file=self._args.motion_file, device=self._device)

    def _init_motion_buffers(self) -> None:
        assert self._motion_lib is not None
        # per-env motion selection and time offset
        self._motion_ids = torch.zeros(self.num_envs, device=self._device, dtype=torch.long)
        self._motion_time_offsets = torch.zeros(
            self.num_envs, device=self._device, dtype=torch.float32
        )

        # reference trajectories (current frame)
        self.ref_root_pos = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_root_rot = torch.zeros(self.num_envs, 4, device=self._device)
        self.ref_root_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_root_ang_vel = torch.zeros(self.num_envs, 3, device=self._device)
        self.ref_dof_pos = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.ref_dof_vel = torch.zeros(self.num_envs, self._robot.dof_dim, device=self._device)
        self.ref_body_pos = torch.zeros(self.num_envs, self._robot.n_links, 3, device=self._device)

        # compact imitation observation (current frame only by default)
        # shape will be (B, 3 + 2 + 3 + 1 + D) = (B, 9 + D)
        self.mimic_obs = torch.zeros(self.num_envs, 9 + self._robot.dof_dim, device=self._device)

        # initialize once
        self.reset_idx(torch.arange(self.num_envs, device=self._device, dtype=torch.long))

    # ---------- Motion utilities ----------
    def _reset_ref_motion(self, envs_idx: torch.Tensor) -> None:
        assert self._motion_lib is not None
        n = len(envs_idx)
        motion_ids = self._motion_lib.sample_motions(n)
        motion_times = self._motion_lib.sample_time(motion_ids)
        self._motion_ids[envs_idx] = motion_ids
        self._motion_time_offsets[envs_idx] = motion_times

        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos_local = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )

        # lift slightly to avoid penetration
        root_pos[:, 2] += 0.05

        # update ref buffers for these envs
        self.ref_root_pos[envs_idx] = root_pos
        self.ref_root_rot[envs_idx] = root_rot
        self.ref_root_vel[envs_idx] = root_vel
        self.ref_root_ang_vel[envs_idx] = root_ang_vel
        self.ref_dof_pos[envs_idx] = dof_pos
        self.ref_dof_vel[envs_idx] = dof_vel
        # convert local body pos to global using ref root pose
        # self.ref_body_pos[envs_idx] = self._local_to_global(root_pos, root_rot, body_pos_local)

    def _get_motion_times(self, envs_idx: torch.Tensor | None = None) -> torch.Tensor:
        if envs_idx is None:
            return self.time_since_reset + self._motion_time_offsets
        return self.time_since_reset[envs_idx] + self._motion_time_offsets[envs_idx]

    def _update_ref_motion(self) -> None:
        assert self._motion_lib is not None
        motion_ids = self._motion_ids
        motion_times = self.time_since_reset + self._motion_time_offsets
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos_local = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        )
        self.ref_root_pos[:] = root_pos
        self.ref_root_rot[:] = root_rot
        self.ref_root_vel[:] = root_vel
        self.ref_root_ang_vel[:] = root_ang_vel
        self.ref_dof_pos[:] = dof_pos
        self.ref_dof_vel[:] = dof_vel
        # self.ref_body_pos[:] = self._local_to_global(root_pos, root_rot, body_pos_local)

    @staticmethod
    def _local_to_global(
        root_pos: torch.Tensor, root_rot: torch.Tensor, local_pos: torch.Tensor
    ) -> torch.Tensor:
        num_envs, num_links, _ = local_pos.shape
        local_flat = local_pos.reshape(num_envs * num_links, 3)
        rot_rep = root_rot[:, None, :].repeat(1, num_links, 1).reshape(num_envs * num_links, 4)
        pos_rep = root_pos[:, None, :].repeat(1, num_links, 1).reshape(num_envs * num_links, 3)
        world = pos_rep + quat_apply(rot_rep, local_flat)
        return world.reshape(num_envs, num_links, 3)

    # ---------- Overrides ----------
    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        # set reference motion first
        # self._reset_ref_motion(envs_idx=envs_idx)

        # initialize robot state close to reference at reset
        default_pos = self.ref_root_pos[envs_idx]
        default_quat = self.ref_root_rot[envs_idx]
        dof_pos = self.ref_dof_pos[envs_idx]
        dof_vel = self.ref_dof_vel[envs_idx] * 0.0
        self.time_since_reset[envs_idx] = 0.0
        self._robot.set_state(
            pos=default_pos, quat=default_quat, dof_pos=dof_pos, envs_idx=envs_idx
        )
        # best-effort zero velocities initially (can be extended to set root/dof velocities if supported)
        _ = dof_vel  # placeholder so linters don't complain

        # reset auxiliary buffers
        self.feet_air_time[envs_idx] = 0.0

    def apply_action(self, action: torch.Tensor) -> None:
        super().apply_action(action=action)
        self.feet_first_contact[:] = (self.feet_air_time > 0.0) * self.feet_contact
        self.feet_air_time += self.dt

    def _pre_step(self) -> None:
        super()._pre_step()

    def update_history(self) -> None:
        super().update_history()
        self.feet_air_time *= 1 - self.feet_contact

    def _update_buffers(self) -> None:
        super()._update_buffers()
        # contacts
        self.feet_contact_force[:] = self.link_contact_forces[:, self._robot.foot_links_idx, 2]
        self.feet_contact[:] = self.feet_contact_force > 1.0
        # update reference for current time
        self._update_ref_motion()
        # build a compact mimic observation (current frame)
        # roll and pitch from current base_quat and ref yaw rate proxy
        # project gravity to get roll/pitch-like signal
        roll_pitch = self.projected_gravity[:, :2]
        # use only yaw component of ref ang vel
        ref_yaw_rate = self.ref_root_ang_vel[:, 2:3]
        self.mimic_obs[:] = torch.cat(
            [
                self.ref_root_pos[:, :3],
                roll_pitch,
                self.ref_root_vel[:, :3],
                ref_yaw_rate,
                self.ref_dof_pos,
            ],
            dim=-1,
        )
