import numpy as np
import torch
import yaml
from genesis.utils import geom as gu

from gs_env.common.bases.base_env import BaseEnv
from gs_env.real.config.schema import OptitrackEnvArgs

from .optitrack.NatNetClient import setup_optitrack

_DEFAULT_DEVICE = torch.device("cpu")


class Real2SimEnv(BaseEnv):
    def __init__(
        self,
        num_envs: int,
        args: OptitrackEnvArgs,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(device=device)

        if num_envs != 1:
            raise ValueError("Real2SimEnv only supports num_envs=1")

        self._num_envs = num_envs
        self._args = args
        self._device = device

        self._client = setup_optitrack(
            server_address=self._args.server_ip,
            client_address=self._args.client_ip,
            use_multicast=self._args.use_multicast,
        )

        self.robot_link_offsets = {}
        with open(self._args.offset_config) as f:
            off = yaml.safe_load(f)
        for name, data in off.items():
            self.robot_link_offsets[name] = {
                "pos": np.array(data["pos"], dtype=np.float32),
                "quat": np.array(data["quat"], dtype=np.float32),
            }

        self._device = device

    def _calculate_tracked_link_by_name(
        self, name: str, pos: np.typing.NDArray[np.float32], quat: np.typing.NDArray[np.float32]
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        """
        Calculate a single tracked link by name.
        """
        if name not in self.robot_link_offsets:
            raise ValueError(f"Tracked link {name} not found!")
        aligned_quat, aligned_pos = self._transform_RT_by(
            quat,
            pos,
            self.robot_link_offsets[name]["quat"],
            self.robot_link_offsets[name]["pos"],
        )
        return aligned_quat, aligned_pos

    def get_tracked_links(
        self,
    ) -> dict[str, tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]]:
        """
        Get all tracked links.
        """
        aligned_poses = {}
        frame = self._client.get_frame()

        for name, (pos, quat) in frame.items():
            if name in self._args.tracked_link_names:
                new_pos, new_quat = self._calculate_tracked_link_by_name(name, pos, quat)
                aligned_poses[name] = (new_pos, new_quat)
        return aligned_poses

    def _transform_RT_by(
        self,
        R1: np.typing.NDArray[np.float32],
        T1: np.typing.NDArray[np.float32],
        R2: np.typing.NDArray[np.float32],
        T2: np.typing.NDArray[np.float32],
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        """
        Apply the offset (R2, T2) to the pose (R1, T1).
        R = R2 * R1
        T = R1 * T2 + T1
        """
        R_out = np.array(gu.transform_quat_by_quat(R1, R2))
        T_out = np.array(gu.transform_by_quat(T2, R1) + T1)
        return R_out, T_out

    def _get_RT_between(
        self,
        R1: np.typing.NDArray[np.float32],
        T1: np.typing.NDArray[np.float32],
        R2: np.typing.NDArray[np.float32],
        T2: np.typing.NDArray[np.float32],
    ) -> tuple[np.typing.NDArray[np.float32], np.typing.NDArray[np.float32]]:
        """
        Get the offset from (R1, T1) to (R2, T2).
        R = R2 * R1^T
        T = R1^T * (T2 - T1)
        """
        R_out = np.array(gu.transform_quat_by_quat(gu.inv_quat(R1), R2))
        T_out = np.array(gu.transform_by_quat(T2 - T1, gu.inv_quat(R1)))
        return R_out, T_out
