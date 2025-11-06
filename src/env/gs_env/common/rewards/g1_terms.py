import torch

from gs_env.common.utils.math_utils import quat_diff, quat_to_angle_axis

from .leggedrobot_terms import (
    ActionLimitPenalty,  # noqa
    ActionRatePenalty,  # noqa
    AngVelXYPenalty,  # noqa
    AngVelZReward,  # noqa
    BaseHeightPenalty,
    DofPosLimitPenalty,  # noqa
    DofVelPenalty,  # noqa
    FeetAirTimePenalty,  # noqa
    FeetAirTimeReward,  # noqa
    FeetContactForceLimitPenalty,
    FeetHeightPenalty,
    FeetZVelocityPenalty,  # noqa
    LinVelXYReward,  # noqa
    LinVelZPenalty,  # noqa
    OrientationPenalty,  # noqa
    StandStillActionRatePenalty,  # noqa
    StandStillFeetContactPenalty,  # noqa
    StandStillReward,  # noqa
    TorquePenalty,  # noqa
)
from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class G1BaseHeightPenalty(BaseHeightPenalty):
    target_height = 0.75


class UpperBodyDofPenalty(RewardTerm):
    """
    Penalize the upper body DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos", "dof_vel")

    def _compute(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, 12:]), dim=-1) - 0.01 * torch.sum(
            torch.square(dof_vel[:, 12:]), dim=-1
        )


class UpperBodyActionPenalty(RewardTerm):
    """
    Penalize the upper body action position.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(action[:, 15:]), dim=-1)


class WaistDofPenalty(RewardTerm):
    """
    Penalize the waist DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [12, 13, 14]]), dim=-1)


class HipYawPenalty(RewardTerm):
    """
    Penalize the hip yaw DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [2, 8]]), dim=-1)


class HipRollPenalty(RewardTerm):
    """
    Penalize the hip roll DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos",)

    def _compute(self, dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, [0, 6]]), dim=-1)


class AnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1)


class StandStillAnkleTorquePenalty(RewardTerm):
    """
    Penalize the ankle torque.

    Args:
        torque: Torque tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("torque", "commands")

    def _compute(self, torque: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(torque[:, [4, 5, 10, 11]]), dim=-1) * (
            torch.norm(commands, dim=1) < 0.1
        )


class G1FeetHeightPenalty(FeetHeightPenalty):
    target_height = 0.2


class G1FeetContactForcePenalty(RewardTerm):
    """
    Penalize the feet contact force.

    Args:
        feet_contact_force: Feet contact force tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("feet_contact_force", "commands")

    def _compute(self, feet_contact_force: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        contact_force_diff = 200 - feet_contact_force.max(dim=-1).values.clamp(max=200)
        contact_force_diff *= torch.norm(commands, dim=1) > 0.1
        return -torch.square(contact_force_diff / 200)


class FeetOrientationPenalty(RewardTerm):
    """
    Penalize the feet orientation.

    Args:
        feet_orientation: Feet orientation tensor of shape (B, 2, 3) where B is the batch size.
    """

    required_keys = ("feet_orientation",)

    def _compute(self, feet_orientation: torch.Tensor) -> torch.Tensor:  # type: ignore
        feet_orientation_deviation = feet_orientation[:, :, :2].square().sum(dim=-1)
        return -feet_orientation_deviation.sum(dim=-1)


class G1FeetContactForceLimitPenalty(FeetContactForceLimitPenalty):
    contact_force_limit = 300.0


class LinVelYPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Y direction.

    Args:
        base_lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel",)

    def _compute(self, base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_lin_vel[:, 1])


class DofPosReward(RewardTerm):
    """
    Reward the DoF position.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        ref_dof_pos: Reference DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_pos", "ref_dof_pos")

    def _compute(self, dof_pos: torch.Tensor, ref_dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # dof pos err weights
        dof_pos_error = torch.square(dof_pos - ref_dof_pos).sum(dim=-1)
        # print("dof_pos_error", dof_pos_error, torch.exp(-dof_pos_error * 0.15))
        return torch.exp(-dof_pos_error * 0.15)


class DofVelReward(RewardTerm):
    """
    Reward the DoF velocity.

    Args:
        dof_vel: DoF velocity tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        ref_dof_vel: Reference DoF velocity tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
    """

    required_keys = ("dof_vel", "ref_dof_vel")

    def _compute(self, dof_vel: torch.Tensor, ref_dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # dof pos err weights
        dof_vel_error = torch.square(dof_vel - ref_dof_vel).sum(dim=-1)
        # print("dof_vel_error", dof_vel_error, torch.exp(-dof_vel_error * 0.01))
        return torch.exp(-dof_vel_error * 0.01)


class BaseHeightReward(RewardTerm):
    """
    Reward the base height.

    Args:
        base_height: Base height tensor of shape (B, 1) where B is the batch size.
        ref_base_height: Reference base height tensor of shape (B, 1) where B is the batch size.
    """

    required_keys = ("base_pos", "ref_base_pos")

    def _compute(self, base_pos: torch.Tensor, ref_base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_height_error = torch.square(base_pos[:, 2] - ref_base_pos[:, 2])
        return torch.exp(-base_height_error * 10)


class BasePosReward(RewardTerm):
    """
    Reward the base position.

    Args:
        base_pos: Base position tensor of shape (B, 3) where B is the batch size.
        ref_base_pos: Reference base position tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_pos", "ref_base_pos")

    def _compute(self, base_pos: torch.Tensor, ref_base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_pos_error = torch.square(base_pos - ref_base_pos).sum(dim=-1)
        # print("base_pos_error", base_pos_error, torch.exp(-base_pos_error * 5))
        return torch.exp(-base_pos_error * 5)


class BaseQuatReward(RewardTerm):
    """
    Reward the base quaternion.

    Args:
        base_quat: Base quaternion tensor of shape (B, 4) where B is the batch size.
        ref_base_quat: Reference base quaternion tensor of shape (B, 4) where B is the batch size.
    """

    required_keys = ("base_quat", "ref_base_quat")

    def _compute(self, base_quat: torch.Tensor, ref_base_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_quat_error = quat_to_angle_axis(quat_diff(base_quat, ref_base_quat)).norm(dim=-1)
        # print("base_quat_error", base_quat_error, torch.exp(-(base_quat_error**2) * 5))
        return torch.exp(-(base_quat_error**2) * 5)


class BaseLinVelReward(RewardTerm):
    """
    Reward the base linear velocity.

    Args:
        base_lin_vel: Base linear velocity tensor of shape (B, 3) where B is the batch size.
        ref_base_lin_vel: Reference base linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_lin_vel", "ref_base_lin_vel")

    def _compute(self, base_lin_vel: torch.Tensor, ref_base_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_lin_vel_error = torch.square(base_lin_vel - ref_base_lin_vel).sum(dim=-1)
        # print("base_lin_vel_error", base_lin_vel_error, torch.exp(-base_lin_vel_error * 1))
        return torch.exp(-base_lin_vel_error * 1)


class BaseAngVelReward(RewardTerm):
    """
    Reward the base angular velocity.

    Args:
        base_ang_vel: Base angular velocity tensor of shape (B, 3) where B is the batch size.
        ref_base_ang_vel: Reference base angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("base_ang_vel", "ref_base_ang_vel")

    def _compute(self, base_ang_vel: torch.Tensor, ref_base_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        base_ang_vel_error = torch.square(base_ang_vel - ref_base_ang_vel).sum(dim=-1)
        # print("base_ang_vel_error", base_ang_vel_error, torch.exp(-base_ang_vel_error * 1))
        return torch.exp(-base_ang_vel_error * 1)


class TrackingLinkPosReward(RewardTerm):
    """
    Reward the tracking link position.

    Args:
        tracking_link_pos_local_yaw: Tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
        ref_tracking_link_pos_local_yaw: Reference tracking link position tensor of shape (B, N, 3) where B is the batch size and N is the number of tracking links.
    """

    required_keys = ("tracking_link_pos_local_yaw", "ref_tracking_link_pos_local_yaw")

    def _compute(
        self,
        tracking_link_pos_local_yaw: torch.Tensor,
        ref_tracking_link_pos_local_yaw: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore
        tracking_link_pos_error = (
            torch.square(tracking_link_pos_local_yaw - ref_tracking_link_pos_local_yaw)
            .sum(dim=-1)
            .sum(dim=-1)
        )
        return torch.exp(-tracking_link_pos_error * 2)
