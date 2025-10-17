import numpy as np
import torch
from gymnasium import spaces
from transforms3d import quaternions

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.real.unitree.utils.low_state_controller import LowStateCmdHandler
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

_DEFAULT_DEVICE = torch.device("cpu")


class UnitreeLeggedEnv(BaseGymRobot):
    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        action_scale: float = 0.0,
        interactive: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self._args = args
        if interactive:
            self._robot = LowStateCmdHandler(args.robot_args)
        else:
            self._robot = LowStateMsgHandler(args.robot_args)
        self.robot.init()
        self.robot.start()
        self._action_space = spaces.Box(shape=(self.action_dim,), low=-np.inf, high=np.inf)
        self._action_scale = args.robot_args.action_scale * action_scale
        self._device = device

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        # TODO: implement reset to reset_pos
        pass

    def apply_action(self, action: torch.Tensor) -> None:
        action_np = action[0].cpu().numpy()
        target_pos = self.robot.default_dof_pos + action_np * self._action_scale
        self.robot.target_pos = target_pos

    def emergency_stop(self) -> None:
        self.robot.emergency_stop()

    @property
    def is_emergency_stop(self) -> bool:
        return self.robot.is_emergency_stop

    @property
    def robot(self) -> LowStateCmdHandler:
        return self._robot

    @property
    def action_dim(self) -> int:
        return self._robot.num_dof

    @property
    def action_space(self) -> spaces.Box:
        return self._action_space

    @property
    def action_scale(self) -> float:
        return self._action_scale

    @property
    def dof_names(self) -> list[str]:
        return self.robot.dof_names

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.default_dof_pos, device=self._device, dtype=torch.float32)[
            None, :
        ]

    @property
    def dof_pos(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_pos, device=self._device, dtype=torch.float32)[None, :]

    @property
    def dof_vel(self) -> torch.Tensor:
        return torch.tensor(self.robot.joint_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def quat(self) -> torch.Tensor:
        return torch.tensor(self.robot.quat, device=self._device, dtype=torch.float32)[None, :]

    @property
    def projected_gravity(self) -> torch.Tensor:
        projected_gravity = quaternions.rotate_vector(
            v=np.array([0, 0, -1]),
            q=quaternions.qinverse(self.robot.quat),
        )
        return torch.tensor(projected_gravity, device=self._device, dtype=torch.float32)[None, :]

    @property
    def base_ang_vel(self) -> torch.Tensor:
        return torch.tensor(self.robot.ang_vel, device=self._device, dtype=torch.float32)[None, :]

    @property
    def device(self) -> torch.device:
        return self._device
