import json

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler,
    quat_inv,
    quat_mul,
    quat_to_euler,
    quat_to_rotation_6D,
)


class RedisClient:
    def __init__(
        self,
        url: str,
        key: str,
        device: str,
        num_tracking_links: int = 0,
    ) -> None:
        self._r = redis.from_url(url)
        self._key = key
        self._device = device
        self._dof_dim = 29
        # Raw variables (before transformations)
        self.base_pos = torch.zeros(1, 3, device=device)
        self.base_quat = torch.zeros(1, 4, device=device)
        self.base_quat[:, 0] = 1.0
        self.last_base_pos = torch.zeros(1, 3, device=device)
        self.last_base_quat = torch.zeros(1, 4, device=device)
        self.last_base_quat[:, 0] = 1.0
        self.base_lin_vel = torch.zeros(1, 3, device=device)
        self.base_ang_vel = torch.zeros(1, 3, device=device)
        self.base_ang_vel_local = torch.zeros(1, 3, device=device)
        self.dof_pos = torch.zeros(1, self._dof_dim, device=device)
        self.dof_vel = torch.zeros(1, self._dof_dim, device=device)
        self.foot_contact = torch.zeros(1, 0, device=device)
        # Ref variables (after transformations)
        self.ref_dof_pos = torch.zeros(1, self._dof_dim, device=device)
        self.ref_dof_vel = torch.zeros(1, self._dof_dim, device=device)
        self.ref_base_pos = torch.zeros(1, 3, device=device)
        self.ref_base_quat = torch.zeros(1, 4, device=device)
        self.ref_base_euler = torch.zeros(1, 3, device=device)
        self.ref_base_rotation_6D = torch.zeros(1, 6, device=device)
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local = torch.zeros(1, 3, device=device)
        self.ref_base_ang_vel_local = torch.zeros(1, 3, device=device)
        # timestamp variables
        self.base_pos_timestamp = -1
        self.base_quat_timestamp = -1
        self.base_lin_vel_timestamp = -1
        self.base_ang_vel_timestamp = -1
        self.base_ang_vel_local_timestamp = -1
        self.dof_pos_timestamp = -1
        self.dof_vel_timestamp = -1
        self.link_pos_local_timestamp = -1
        self.link_quat_local_timestamp = -1
        self.foot_contact_timestamp = -1

        self.num_tracking_links = num_tracking_links
        self.link_pos_local_yaw = torch.zeros(1, num_tracking_links, 3, device=device)
        self.link_quat_local_yaw = torch.zeros(1, num_tracking_links, 4, device=device)

        # Yaw difference quaternion (stored and applied to all subsequent updates)
        self._yaw_diff_quat = torch.zeros(1, 4, device=device)
        self._yaw_diff_quat[:, 0] = 1.0
        # Motion obs element selection (None or empty => compute none by default)
        self._motion_obs_elements: set[str] | None = None

    def _zero_all(self) -> None:
        # Zero raw variables
        self.base_pos.zero_()
        self.base_quat.zero_()
        self.base_quat[:, 0] = 1.0
        self.last_base_pos.zero_()
        self.last_base_quat.zero_()
        self.last_base_quat[:, 0] = 1.0
        self.base_lin_vel.zero_()
        self.base_ang_vel.zero_()
        self.base_ang_vel_local.zero_()
        self.dof_pos.zero_()
        self.dof_vel.zero_()
        self.foot_contact.zero_()
        # Zero ref variables
        self.ref_dof_pos.zero_()
        self.ref_dof_vel.zero_()
        self.ref_base_pos.zero_()
        self.ref_base_quat.zero_()
        self.ref_base_quat[:, 0] = 1.0
        self.ref_base_euler.zero_()
        self.ref_base_rotation_6D.zero_()
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local.zero_()
        self.ref_base_ang_vel_local.zero_()

    def _fit_dim(self, data: list[float], dim: int) -> torch.Tensor:
        out = torch.zeros(1, dim, device=self._device)
        if len(data) > 0:
            n = min(dim, len(data))
            out[0, :n] = torch.tensor(data[:n], dtype=torch.float32, device=self._device)
        return out

    def _fit_link_data(self, data: list[float], n_links: int, dim_per_link: int) -> torch.Tensor:
        """Reshape flattened link data into (1, n_links, dim_per_link) tensor."""
        out = torch.zeros(1, n_links, dim_per_link, device=self._device)
        if len(data) > 0 and n_links > 0:
            expected_size = n_links * dim_per_link
            n = min(expected_size, len(data))
            reshaped = torch.tensor(data[:n], dtype=torch.float32, device=self._device).view(
                -1, dim_per_link
            )
            actual_links = min(n_links, reshaped.shape[0])
            out[0, :actual_links, :] = reshaped[:actual_links, :]
        return out

    def _get_field(self, field_name: str, default: list[float]) -> list[float]:
        """Get a field from Redis as a JSON-decoded list."""
        # Try both formats: with :motion: prefix (new format) and without (old format)
        raw = self._r.get(f"{self._key}:motion:{field_name}")
        if raw is None:
            raw = self._r.get(f"{self._key}:{field_name}")
        if raw is None:
            return default
        s = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
        return json.loads(s)

    def _get_timestamp(self, field_name: str) -> int:
        """Get a timestamp from Redis for a given field."""
        # Try both formats: with :timestamp: prefix (new format) and without (old format)
        raw = self._r.get(f"{self._key}:timestamp:{field_name}")
        if raw is None:
            # Fallback: check if field exists at all
            return -1
        try:
            if isinstance(raw, bytes | bytearray):
                return int(raw.decode("utf-8"))
            return int(str(raw))
        except (ValueError, TypeError):
            return -1

    def update(self) -> None:
        try:
            # Update base_pos if timestamp changed
            new_timestamp = self._get_timestamp("base_pos")
            if new_timestamp != self.base_pos_timestamp:
                # Store previous value before updating
                self.last_base_pos.copy_(self.base_pos)
                base_pos = torch.tensor(
                    self._get_field("base_pos", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                self.base_pos = base_pos  # Store raw value
                self.ref_base_pos = quat_apply(self._yaw_diff_quat, base_pos)
                self.base_pos_timestamp = new_timestamp

            # Update base_quat if timestamp changed
            new_timestamp = self._get_timestamp("base_quat")
            if new_timestamp != self.base_quat_timestamp:
                # Store previous value before updating
                self.last_base_quat.copy_(self.base_quat)
                base_quat = torch.tensor(
                    self._get_field("base_quat", [1.0, 0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 4)
                self.base_quat = base_quat  # Store raw value
                self.ref_base_quat = quat_mul(self._yaw_diff_quat, base_quat)
                # self.ref_base_quat = base_quat
                self.base_quat_timestamp = new_timestamp
                # Update derived quantities when quat changes
                self.ref_base_euler = quat_to_euler(self.ref_base_quat)
                self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)

            # Update base_lin_vel if timestamp changed
            new_timestamp = self._get_timestamp("base_lin_vel")
            if new_timestamp != self.base_lin_vel_timestamp:
                base_lin_vel = torch.tensor(
                    self._get_field("base_lin_vel", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                self.base_lin_vel = base_lin_vel  # Store raw value
                # Apply yaw_diff_quat first (equivalent to base_yaw_offset_quat in motion_env)
                base_lin_vel = quat_apply(self._yaw_diff_quat, base_lin_vel)
                # Convert to local frame using ref_base_quat (equivalent to batched_global_to_local)
                inv_quat = quat_inv(self.ref_base_quat)
                self.ref_base_lin_vel_local = quat_apply(inv_quat, base_lin_vel)
                self.base_lin_vel_timestamp = new_timestamp

            # Update base_ang_vel if timestamp changed
            new_timestamp = self._get_timestamp("base_ang_vel")
            if new_timestamp != self.base_ang_vel_timestamp:
                base_ang_vel = torch.tensor(
                    self._get_field("base_ang_vel", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                self.base_ang_vel = base_ang_vel  # Store raw value
                # Apply yaw_diff_quat first (equivalent to base_yaw_offset_quat in motion_env)
                base_ang_vel = quat_apply(self._yaw_diff_quat, base_ang_vel)
                # Convert to local frame using ref_base_quat (equivalent to batched_global_to_local)
                inv_quat = quat_inv(self.ref_base_quat)
                self.ref_base_ang_vel_local = quat_apply(inv_quat, base_ang_vel)
                self.base_ang_vel_timestamp = new_timestamp

            # Update base_ang_vel_local if timestamp changed (from motion library, used directly)
            new_timestamp = self._get_timestamp("base_ang_vel_local")
            if new_timestamp != self.base_ang_vel_local_timestamp:
                base_ang_vel_local = torch.tensor(
                    self._get_field("base_ang_vel_local", [0.0, 0.0, 0.0]),
                    dtype=torch.float32,
                    device=self._device,
                ).view(1, 3)
                self.base_ang_vel_local = base_ang_vel_local  # Store raw value
                # Use directly without transformation (as in motion_env.py line 627)
                self.ref_base_ang_vel_local = base_ang_vel_local
                self.base_ang_vel_local_timestamp = new_timestamp

            # Update dof_pos if timestamp changed
            new_timestamp = self._get_timestamp("dof_pos")
            if new_timestamp != self.dof_pos_timestamp:
                dof_pos = self._fit_dim(self._get_field("dof_pos", []), self._dof_dim)
                self.dof_pos = dof_pos  # Store raw value
                self.ref_dof_pos = dof_pos
                self.dof_pos_timestamp = new_timestamp

            # Update dof_vel if timestamp changed
            new_timestamp = self._get_timestamp("dof_vel")
            if new_timestamp != self.dof_vel_timestamp:
                dof_vel = self._fit_dim(self._get_field("dof_vel", []), self._dof_dim)
                self.dof_vel = dof_vel  # Store raw value
                self.ref_dof_vel = dof_vel
                self.dof_vel_timestamp = new_timestamp

            # Parse link positions and quaternions if available
            if self.num_tracking_links > 0:
                # Update link_pos_local if timestamp changed
                new_timestamp = self._get_timestamp("link_pos_local")
                if new_timestamp != self.link_pos_local_timestamp:
                    link_pos_local = self._fit_link_data(
                        self._get_field("link_pos_local", []), self.num_tracking_links, 3
                    )
                    self.link_pos_local = link_pos_local
                    # link_pos_local is already in local-yaw frame, so set link_pos_local_yaw
                    self.link_pos_local_yaw = link_pos_local
                    self.link_pos_local_timestamp = new_timestamp

                # Update link_quat_local if timestamp changed
                new_timestamp = self._get_timestamp("link_quat_local")
                if new_timestamp != self.link_quat_local_timestamp:
                    link_quat_local = self._fit_link_data(
                        self._get_field("link_quat_local", []), self.num_tracking_links, 4
                    )
                    self.link_quat_local = link_quat_local  # Store raw value
                    # link_quat_local is already in local-yaw frame, so set link_quat_local_yaw
                    self.link_quat_local_yaw = link_quat_local
                    self.link_quat_local_timestamp = new_timestamp

            # Update foot_contact if timestamp changed (used directly without transformation)
            new_timestamp = self._get_timestamp("foot_contact")
            if new_timestamp != self.foot_contact_timestamp:
                foot_contact_data = self._get_field("foot_contact", [])
                if len(foot_contact_data) > 0:
                    # Resize buffers if needed
                    n_feet = len(foot_contact_data)
                    if self.foot_contact.shape[1] != n_feet:
                        self.foot_contact = torch.zeros(1, n_feet, device=self._device)
                        self.ref_foot_contact = torch.zeros(1, n_feet, device=self._device)
                    foot_contact = torch.tensor(
                        foot_contact_data, dtype=torch.float32, device=self._device
                    ).view(1, n_feet)
                    self.foot_contact = foot_contact  # Store raw value
                    # Use directly without transformation (as in motion_env.py line 638)
                    self.ref_foot_contact = foot_contact
                    self.foot_contact_timestamp = new_timestamp

        except Exception as e:
            print(f"Error updating Redis client: {e}")
            self._zero_all()

    @staticmethod
    def _batched_global_to_local(base_quat: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        """Convert global vectors to local frame using quaternion.

        Args:
            base_quat: Quaternion tensor of shape (B, 4)
            global_vec: Global vector tensor of shape (B, L, 3) or (B, L, 4)

        Returns:
            Local vector tensor of same shape as global_vec
        """
        assert base_quat.shape[0] == global_vec.shape[0]
        global_vec_shape = global_vec.shape
        global_vec = global_vec.reshape(global_vec_shape[0], -1, global_vec_shape[-1])
        B, L, D = global_vec.shape
        global_flat = global_vec.reshape(B * L, D)
        quat_rep = base_quat[:, None, :].repeat(1, L, 1).reshape(B * L, 4)
        if D == 3:
            local_flat = quat_apply(quat_inv(quat_rep), global_flat)
        elif D == 4:
            local_flat = quat_mul(quat_inv(quat_rep), global_flat)
        else:
            raise ValueError(
                f"Global vector shape must be (B, L, 3) or (B, L, 4), but got {global_flat.shape}"
            )
        return local_flat.reshape(global_vec_shape)

    def compute_motion_obs(self) -> torch.Tensor:
        """Compute motion observation with 1 future step.

        Builds a subset of motion_obs similar to MotionEnv._build_motion_obs_from_dict.
        Uses current ref_ values as the future frame and last_base_* as current.
        Included elements are controlled via set_motion_obs_elements(); default: none.

        Returns:
            Motion observation tensor of shape (1, M).
        """
        # If no elements selected, return empty observation
        if not self._motion_obs_elements:
            return torch.zeros(1, 0, device=self._device)

        # Compute yaw quaternion from last_base_quat
        quat_yaw = quat_from_angle_axis(
            quat_to_euler(self.last_base_quat)[:, 2],
            torch.tensor([0, 0, 1], device=self._device, dtype=torch.float32),
        )

        motion_obs_list: list[torch.Tensor] = []

        # base_pos difference (future - current) in local-yaw frame
        if "base_pos" in self._motion_obs_elements:
            motion_obs_list.append(
                self._batched_global_to_local(
                    quat_yaw, (self.ref_base_pos.unsqueeze(1) - self.last_base_pos[:, None, :])
                ).reshape(1, -1)
            )

        # base_quat in local-yaw frame and difference from current base_quat
        if "base_quat" in self._motion_obs_elements:
            qy = quat_yaw[:, None, :].repeat(1, 1, 1)
            future_base_quat = self.ref_base_quat.unsqueeze(1)
            motion_obs_list.append(
                quat_to_rotation_6D(quat_mul(quat_inv(qy), future_base_quat)).reshape(1, -1)
            )
            motion_obs_list.append(
                quat_to_rotation_6D(
                    quat_mul(
                        quat_inv(self.last_base_quat[:, None, :].repeat(1, 1, 1)), future_base_quat
                    )
                ).reshape(1, -1)
            )

        # base_lin_vel in local-yaw frame
        if "base_lin_vel" in self._motion_obs_elements:
            motion_obs_list.append(
                self._batched_global_to_local(
                    quat_yaw,
                    quat_apply(self.ref_base_quat, self.ref_base_lin_vel_local).unsqueeze(1),
                ).reshape(1, -1)
            )

        # base_ang_vel in local-yaw frame
        if "base_ang_vel" in self._motion_obs_elements:
            motion_obs_list.append(
                self._batched_global_to_local(
                    quat_yaw,
                    quat_apply(self.ref_base_quat, self.ref_base_ang_vel_local).unsqueeze(1),
                ).reshape(1, -1)
            )

        # base_ang_vel_local (already local)
        if "base_ang_vel_local" in self._motion_obs_elements:
            motion_obs_list.append(self.ref_base_ang_vel_local.unsqueeze(1).reshape(1, -1))

        # dof_pos
        if "dof_pos" in self._motion_obs_elements:
            motion_obs_list.append(self.ref_dof_pos.unsqueeze(1).reshape(1, -1))

        # dof_vel (scaled by 0.1)
        if "dof_vel" in self._motion_obs_elements:
            motion_obs_list.append((0.1 * self.ref_dof_vel.unsqueeze(1)).reshape(1, -1))

        # link_pos_local (tracking links only)
        if (
            "link_pos_local" in self._motion_obs_elements
            and self.num_tracking_links > 0
            and self.link_pos_local_yaw.shape[1] > 0
        ):
            motion_obs_list.append(self.link_pos_local_yaw.unsqueeze(1).reshape(1, -1))

        # link_quat_local (tracking links only, converted to 6D rotation)
        if (
            "link_quat_local" in self._motion_obs_elements
            and self.num_tracking_links > 0
            and self.link_quat_local_yaw.shape[1] > 0
        ):
            motion_obs_list.append(
                quat_to_rotation_6D(self.link_quat_local_yaw.unsqueeze(1)).reshape(1, -1)
            )

        # foot_contact
        if "foot_contact" in self._motion_obs_elements and self.foot_contact.shape[1] > 0:
            motion_obs_list.append(self.foot_contact.unsqueeze(1).reshape(1, -1))

        return torch.cat(motion_obs_list, dim=-1)

    def update_quat(self, quat: torch.Tensor) -> None:
        """Compute and store yaw difference from input quaternion.
        The stored yaw difference will be applied to all subsequent updates.

        Args:
            quat: Input quaternion tensor of shape (1, 4) in (w, x, y, z) format.
        """
        # Extract yaw from input quaternion
        input_euler = quat_to_euler(quat)
        input_yaw = input_euler[:, 2]  # yaw is the third element (z-axis rotation)

        # Extract yaw from current ref_base_quat
        current_euler = quat_to_euler(self.ref_base_quat)
        current_yaw = current_euler[:, 2]

        # Compute yaw difference
        yaw_diff = input_yaw - current_yaw

        # Create a quaternion representing only the yaw rotation (around z-axis)
        # Euler angles: (roll, pitch, yaw) - we only want yaw rotation
        yaw_only_euler = torch.zeros(1, 3, device=self._device)
        yaw_only_euler[:, 2] = yaw_diff  # only set yaw component
        self._yaw_diff_quat = quat_from_euler(yaw_only_euler)

        # Apply yaw difference to current ref_base_quat
        self.ref_base_quat = quat_mul(self._yaw_diff_quat, self.ref_base_quat)

        # Update derived quantities
        self.ref_base_euler = quat_to_euler(self.ref_base_quat)
        self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)
        return

    def set_motion_obs_elements(self, elements: list[str] | None) -> None:
        """Select which elements to include in compute_motion_obs.

        Args:
            elements: List of term names among:
                ['base_pos','base_quat','base_lin_vel','base_ang_vel',
                 'base_ang_vel_local','dof_pos','dof_vel',
                 'link_pos_local','link_quat_local','foot_contact'].
                 If None or empty, no elements are computed (empty observation).
        """
        valid = {
            "base_pos",
            "base_quat",
            "base_lin_vel",
            "base_ang_vel",
            "base_ang_vel_local",
            "dof_pos",
            "dof_vel",
            "link_pos_local",
            "link_quat_local",
            "foot_contact",
        }
        if elements is None or len(elements) == 0:
            self._motion_obs_elements = None
            return
        filtered = [e for e in elements if e in valid]
        self._motion_obs_elements = set(filtered)
