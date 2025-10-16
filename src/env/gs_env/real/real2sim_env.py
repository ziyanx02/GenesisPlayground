import torch

from gs_env.common.bases.base_robot import BaseGymRobot
from gs_env.real.unitree.utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.robots.config.schema import HumanoidRobotArgs

_DEFAULT_DEVICE = torch.device("cpu")


class Real2SimLeggedEnv(BaseGymRobot):
    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self._args = args
        if type(args.robot_args) is not HumanoidRobotArgs:
            raise ValueError("Real2SimLeggedEnv only supports HumanoidRobotArgs")

        self._low_state_handler = LowStateMsgHandler(args.robot_args)
        self._low_state_handler.init()

        self._device = device

    def reset(self, envs_idx: torch.IntTensor | None = None) -> None:
        pass
