import importlib
import platform

import genesis as gs
import torch

from gs_env.common.bases.base_env import BaseEnv

#
from gs_env.common.utils.math_utils import (
    quat_from_euler,
    quat_mul,
)
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv
from gs_env.sim.robots.leggedrobots import G1Robot
from gs_env.sim.scenes import CustomScene

_DEFAULT_DEVICE = torch.device("cpu")


class CustomEnv(LeggedRobotEnv):
    """
    Custom environment to add arbitrary objects in the scene.
    """

    def __init__(
        self,
        args: LeggedRobotEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        BaseEnv.__init__(self, device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._refresh_visualizer = False if platform.system() == "Darwin" else True
        self._args = args
        self._eval_mode = eval_mode

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))

        # == setup the scene ==
        self._scene = CustomScene(
            num_envs=self._num_envs,
            device=self.device,
            args=args.scene_args,  # type: ignore
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
            env_spacing=(0.0, 0.0),
        )

        # == setup the robot ==
        self._robot = G1Robot(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self.device,
        )

        # == set up camera ==
        self._floating_camera = self._scene.scene.add_camera(
            res=(480, 480),
            fov=40,
            GUI=False,
        )

        # == build the scene ==
        self._scene.build()
        self._max_sim_time = 20.0  # seconds

        # init buffers
        self._init()

        # == setup reward scalars and functions ==
        dt = self._scene.scene.dt
        self._reward_functions = {}
        module_name = f"gs_env.common.rewards.{self._args.reward_term}_terms"
        module = importlib.import_module(module_name)
        for key in args.reward_args.keys():
            reward_func = getattr(module, key, None)
            if reward_func is None:
                raise ValueError(f"Reward {key} not found in rewards module.")
            self._reward_functions[key] = reward_func(scale=args.reward_args[key] * dt)

        # reset the environment
        self.reset()

    def _init(self) -> None:
        super()._init()

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        default_pos = self._robot.default_pos[None, :].repeat(len(envs_idx), 1)
        default_quat = self._robot.default_quat[None, :].repeat(len(envs_idx), 1)
        default_dof_pos = self._robot.default_dof_pos[None, :].repeat(len(envs_idx), 1)
        random_euler = torch.zeros((len(envs_idx), 3), device=self.device)
        random_euler[:, :2] = 0.0
        random_euler[:, 2] = -3.141592653589793 / 2
        quat = quat_from_euler(random_euler)
        quat = quat_mul(quat, default_quat)
        dof_pos = default_dof_pos
        self.time_since_reset[envs_idx] = 0.0
        self._robot.set_state(pos=default_pos, quat=quat, dof_pos=dof_pos, envs_idx=envs_idx)
