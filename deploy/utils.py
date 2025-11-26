import json

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
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
        link_names: list[str] | None = None,
        tracking_link_names: list[str] | None = None,
    ) -> None:
        self._r = redis.from_url(url)
        self._key = key
        self._device = device
        self._dof_dim = 29
        self.ref_dof_pos = torch.zeros(1, self._dof_dim, device=device)
        self.ref_dof_vel = torch.zeros(1, self._dof_dim, device=device)
        self.ref_base_pos = torch.zeros(1, 3, device=device)
        self.ref_base_quat = torch.zeros(1, 4, device=device)
        self.ref_base_euler = torch.zeros(1, 3, device=device)
        self.ref_base_rotation_6D = torch.zeros(1, 6, device=device)
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local = torch.zeros(1, 3, device=device)
        self.ref_base_ang_vel_local = torch.zeros(1, 3, device=device)

        # Link tracking setup
        self._link_names = link_names or []
        self._tracking_link_names = tracking_link_names or []
        self._n_links = len(self._link_names) if self._link_names else 0
        self._tracking_link_idx_local: list[int] = []
        if self._link_names and self._tracking_link_names:
            self._tracking_link_idx_local = [
                self._link_names.index(name) for name in self._tracking_link_names if name in self._link_names
            ]

        # Initialize link buffers
        self.ref_link_pos_local_yaw = torch.zeros(1, self._n_links, 3, device=device)
        self.ref_link_quat_local_yaw = torch.zeros(1, self._n_links, 4, device=device)
        self.ref_tracking_link_pos_local_yaw = torch.zeros(
            1, len(self._tracking_link_idx_local), 3, device=device
        )
        self.ref_tracking_link_quat_local_yaw = torch.zeros(
            1, len(self._tracking_link_idx_local), 4, device=device
        )

        # Yaw difference quaternion (stored and applied to all subsequent updates)
        self._yaw_diff_quat = torch.zeros(1, 4, device=device)
        self._yaw_diff_quat[:, 0] = 1.0

    def _zero_all(self) -> None:
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
        self.ref_link_pos_local_yaw.zero_()
        self.ref_link_quat_local_yaw.zero_()
        self.ref_tracking_link_pos_local_yaw.zero_()
        self.ref_tracking_link_quat_local_yaw.zero_()

    def _fit_dim(self, data: list[float], dim: int) -> torch.Tensor:
        out = torch.zeros(1, dim, device=self._device)
        if len(data) > 0:
            n = min(dim, len(data))
            out[0, :n] = torch.tensor(data[:n], dtype=torch.float32, device=self._device)
        return out

    def _fit_link_data(
        self, data: list[float], n_links: int, dim_per_link: int
    ) -> torch.Tensor:
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

    def update(self) -> None:
        try:
            raw = self._r.get(self._key)
            if raw is None:
                self._zero_all()
                return
            s = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
            msg = json.loads(s)

            base_pos = torch.tensor(
                msg.get("base_pos", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self._device
            ).view(1, 3)
            base_quat = torch.tensor(
                msg.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=torch.float32, device=self._device
            ).view(1, 4)
            base_lin_vel = torch.tensor(
                msg.get("base_lin_vel", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self._device
            ).view(1, 3)
            base_ang_vel = torch.tensor(
                msg.get("base_ang_vel", [0.0, 0.0, 0.0]), dtype=torch.float32, device=self._device
            ).view(1, 3)
            dof_pos = self._fit_dim(msg.get("dof_pos", []), self._dof_dim)
            dof_vel = self._fit_dim(msg.get("dof_vel", []), self._dof_dim)

            inv_quat = quat_inv(base_quat)
            self.ref_base_pos = quat_apply(self._yaw_diff_quat, base_pos)
            self.ref_base_quat = quat_mul(self._yaw_diff_quat, base_quat)
            self.ref_dof_pos = dof_pos
            self.ref_dof_vel = dof_vel
            self.ref_base_euler = quat_to_euler(self.ref_base_quat)
            self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)
            # Recompute inv_quat after applying yaw_diff
            inv_quat = quat_inv(self.ref_base_quat)
            self.ref_base_lin_vel_local = quat_apply(inv_quat, base_lin_vel)
            self.ref_base_ang_vel_local = quat_apply(inv_quat, base_ang_vel)

            # Parse link positions and quaternions if available
            if self._n_links > 0:
                link_pos_local = self._fit_link_data(
                    msg.get("link_pos_local", []), self._n_links, 3
                )
                link_quat_local = self._fit_link_data(
                    msg.get("link_quat_local", []), self._n_links, 4
                )
                self.ref_link_pos_local_yaw = link_pos_local
                self.ref_link_quat_local_yaw = link_quat_local

                # Extract tracking links if indices are available
                if len(self._tracking_link_idx_local) > 0:
                    self.ref_tracking_link_pos_local_yaw = self.ref_link_pos_local_yaw[
                        :, self._tracking_link_idx_local, :
                    ]
                    self.ref_tracking_link_quat_local_yaw = self.ref_link_quat_local_yaw[
                        :, self._tracking_link_idx_local, :
                    ]
        except Exception as e:
            print(f"Error updating Redis client: {e}")
            self._zero_all()

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
        print(yaw_only_euler)
        
        # Apply yaw difference to current ref_base_quat
        self.ref_base_quat = quat_mul(self._yaw_diff_quat, self.ref_base_quat)
        
        # Update derived quantities
        self.ref_base_euler = quat_to_euler(self.ref_base_quat)
        self.ref_base_rotation_6D = quat_to_rotation_6D(self.ref_base_quat)
