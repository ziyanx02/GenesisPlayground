import numpy as np
import torch
from gymnasium import spaces
from transforms3d import quaternions

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler  # noqa
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

_DEFAULT_DEVICE = torch.device("cpu")


class UnitreeLeggedEnv(BaseGymRobot):
    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        action_scale: float = 0.0,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self._args = args
        self._low_state_controller = LowStateCmdHandler(args.robot_args)
        self.controller.init()
        self.controller.start()
        self._action_space = spaces.Box(shape=(self.action_dim,), low=-np.inf, high=np.inf)
        self._action_scale = args.robot_args.action_scale * action_scale
        self._device = device

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        # TODO: implement reset to reset_pos
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        action_np = action[0].cpu().numpy()
        target_pos = self.controller.default_pos + action_np * self._action_scale
        self.controller.target_pos = target_pos

    def emergency_stop(self) -> None:
        self.controller.emergency_stop()

    @property
    def is_emergency_stop(self) -> bool:
        return self.controller.is_emergency_stop

    @property
    def controller(self) -> LowStateCmdHandler:
        return self._low_state_controller

    @property
    def action_dim(self) -> int:
        return self._low_state_controller.num_dof

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def action_scale(self) -> float:
        return self._action_scale

    @property
    def dof_names(self) -> list[str]:
        return self.controller.dof_names

    @property
    def dof_pos(self) -> torch.Tensor:
        dof_pos = self.controller.joint_pos - self.controller.default_pos
        return torch.tensor(dof_pos, device=self._device)[None, :]

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.tensor(self.controller.joint_vel, device=self._device)[None, :]

    @property
    def projected_gravity(self) -> torch.Tensor:
        projected_gravity = quaternions.rotate_vector(
            v=np.array([0, 0, -1]),
            q=quaternions.qinverse(self.controller.quat),
        )
        return torch.tensor(projected_gravity, device=self._device)[None, :]

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return torch.tensor(self.controller.ang_vel, device=self._device)[None, :]

    @property
    def device(self) -> torch.device:
        return self._device
