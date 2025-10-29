import json

import redis
import torch
from gs_env.common.utils.math_utils import (
    quat_apply,
    quat_inv,
    quat_to_euler,
    quat_to_rotation_6D,
)


class RedisClient:
    def __init__(self, url: str, key: str, device: str) -> None:
        self._r = redis.from_url(url)
        self._key = key
        self._device = device
        self._dof_dim = 29
        self.ref_dof_pos = torch.zeros(1, self._dof_dim, device=device)
        self.ref_dof_vel = torch.zeros(1, self._dof_dim, device=device)
        self.ref_base_euler = torch.zeros(1, 3, device=device)
        self.ref_base_rotation_6D = torch.zeros(1, 6, device=device)
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local = torch.zeros(1, 3, device=device)
        self.ref_base_ang_vel_local = torch.zeros(1, 3, device=device)

    def _zero_all(self) -> None:
        self.ref_dof_pos.zero_()
        self.ref_dof_vel.zero_()
        self.ref_base_euler.zero_()
        self.ref_base_rotation_6D.zero_()
        self.ref_base_rotation_6D[:, [0, 4]] = 1.0
        self.ref_base_lin_vel_local.zero_()
        self.ref_base_ang_vel_local.zero_()
        self.ref_base_rotation_6D.zero_()

    def _fit_dim(self, data: list[float], dim: int) -> torch.Tensor:
        out = torch.zeros(1, dim, device=self._device)
        if len(data) > 0:
            n = min(dim, len(data))
            out[0, :n] = torch.tensor(data[:n], dtype=torch.float32, device=self._device)
        return out

    def update(self) -> None:
        try:
            raw = self._r.get(self._key)
            if raw is None:
                self._zero_all()
                return
            s = raw.decode("utf-8") if isinstance(raw, bytes | bytearray) else str(raw)
            msg = json.loads(s)

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
            self.ref_dof_pos = dof_pos
            self.ref_dof_vel = dof_vel
            self.ref_base_euler = quat_to_euler(base_quat)
            self.ref_base_rotation_6D = quat_to_rotation_6D(base_quat)
            self.ref_base_lin_vel_local = quat_apply(inv_quat, base_lin_vel)
            self.ref_base_ang_vel_local = quat_apply(inv_quat, base_ang_vel)
        except Exception:
            self._zero_all()
