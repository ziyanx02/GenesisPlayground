import torch

from .leggedrobot_terms import (
    ActionLimitPenalty,  # noqa
    ActionRatePenalty,  # noqa
    AngVelXYPenalty,  # noqa
    AngVelZReward,  # noqa
    BaseHeightPenalty,
    DofPosLimitPenalty,  # noqa
    LinVelXYReward,  # noqa
    LinVelZPenalty,  # noqa
    OrientationPenalty,  # noqa
    TorquePenalty,  # noqa
    FeetAirTimeReward,  # noqa
)
from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.7


class UpperBodyDofPenalty(RewardTerm):
    """
    Penalize the upper body DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, 12:]), dim=-1)


class HipYawPenalty(RewardTerm):
    """
    Penalize the hip yaw DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [2, 8]]), dim=-1)


class AnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1)
