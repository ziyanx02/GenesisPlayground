import torch

from .reward_terms import RewardTerm


### ---- Reward Terms ---- ###
class LinVelXYReward(RewardTerm):
    """
    Reward the linear velocity in the X and Y directions.

    Args:
        lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("lin_vel", "commands")

    def _compute(self, lin_vel: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.exp(-torch.sum(torch.square(lin_vel[:, :2] - commands[:, :2]), dim=-1) / 0.25)


class AngVelZReward(RewardTerm):
    """
    Reward the angular velocity in the Z direction.

    Args:
        ang_vel: Angular velocity tensor of shape (B, 3) where B is the batch size.
        command_ang_vel: Commanded angular velocity tensor of shape (B,) representing desired yaw rate.
    """

    required_keys = ("ang_vel", "commands")

    def _compute(self, ang_vel: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.exp(-torch.square(ang_vel[:, 2] - commands[:, 2]) / 0.25)


class LinVelZPenalty(RewardTerm):
    """
    Penalize the linear velocity in the Z direction.

    Args:
        lin_vel: Linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("lin_vel",)

    def _compute(self, lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(lin_vel[:, 2])


class AngVelXYPenalty(RewardTerm):
    """
    Penalize the angular velocity in the X and Y directions.

    Args:
        ang_vel: Angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("ang_vel",)

    def _compute(self, ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(ang_vel[:, :2]), dim=-1)


class OrientationPenalty(RewardTerm):
    """
    Penalize the orientation deviation from upright.

    Args:
        root_quat: Root orientation quaternion tensor of shape (B, 4) where B is the batch size.
    """

    required_keys = ("projected_gravity",)

    def _compute(self, projected_gravity: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)


class BaseHeightPenalty(RewardTerm):
    """
    Penalize the deviation of the base height from a target height.

    Args:
        base_height: Base height tensor of shape (B,) where B is the batch size.
    """

    required_keys = ("base_pos",)
    target_height: float = 1.0

    def _compute(self, base_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.square(base_pos[:, 2] - self.target_height)


class ActionRatePenalty(RewardTerm):
    """
    Penalize the action rate by its squared L2 norm.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
        prev_action: Previous action tensor of shape (B, D).
    """

    required_keys = ("action", "last_action")

    def _compute(self, action: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum((action - last_action) ** 2, dim=-1)


class TorquePenalty(RewardTerm):
    """
    Penalize the torque by its squared L2 norm.

    Args:
        joint_torques: Joint torque tensor of shape (B, D) where B is the batch size and D is the number of joints.
    """

    required_keys = ("torque",)

    def _compute(self, torque: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torque**2, dim=-1)


class DofPosLimitPenalty(RewardTerm):
    """
    Penalize the degree of freedom (DoF) position limit violations.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        dof_pos_limits: DoF position limits tensor of shape (D, 2) where each row contains [min, max] limits for a DoF.
    """

    required_keys = ("dof_pos", "dof_pos_limits")

    def _compute(self, dof_pos: torch.Tensor, dof_pos_limits: torch.Tensor) -> torch.Tensor:  # type: ignore
        out_of_limits = -(dof_pos - dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (dof_pos - dof_pos_limits[:, 1]).clip(min=0.0)  # upper limit
        return torch.sum(out_of_limits, dim=1)


class ActionLimitPenalty(RewardTerm):
    """
    Penalize the action limit violations.

    Args:
        action: Action tensor of shape (B, D) where B is the batch size and D is the action dimension.
        action_limits: Action limits tensor of shape (D, 2) where each row contains [min, max] limits for an action dimension.
    """

    required_keys = ("action",)

    def _compute(self, action: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.sum(torch.square(torch.abs(action).clip(min=8) - 8), dim=1)

class TeleopDofPos(RewardTerm):
    """
    Reward the similarity between the current and target DoF positions for all joints.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_pos: Target DoF position tensor of shape (B, D).
    """
    
    required_keys = ("dof_pos", "target_dof_pos")

    def _compute(self, dof_pos: torch.Tensor, target_dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, :] - target_dof_pos[:, :]), dim=1)

class TeleopDofPosSelected(RewardTerm):
    """
    Reward the similarity between the current and target DoF positions for selected joints.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_pos: Target DoF position tensor of shape (B, D).
        dof_pos_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_pos", "target_dof_pos", "dof_pos_idxs")

    def _compute(self, dof_pos: torch.Tensor, target_dof_pos: torch.Tensor, dof_pos_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, dof_pos_idxs] - target_dof_pos[:, dof_pos_idxs]), dim=1)

class TeleopDofPosLower(RewardTerm):
    """
    Reward the similarity between the current and target DoF positions for lower joints.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_pos: Target DoF position tensor of shape (B, D).
        lower_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_pos", "target_dof_pos", "lower_idxs")

    def _compute(self, dof_pos: torch.Tensor, target_dof_pos: torch.Tensor, lower_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, lower_idxs] - target_dof_pos[:, lower_idxs]), dim=1)

class TeleopDofPosUpper(RewardTerm):
    """
    Reward the similarity between the current and target DoF positions for upper joints.

    Args:
        dof_pos: DoF position tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_pos: Target DoF position tensor of shape (B, D).
        upper_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_pos", "target_dof_pos", "upper_idxs")

    def _compute(self, dof_pos: torch.Tensor, target_dof_pos: torch.Tensor, upper_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_pos[:, upper_idxs] - target_dof_pos[:, upper_idxs]), dim=1)
    
class TeleopDofVelAll(RewardTerm):
    """
    Reward the similarity between the current and target DoF velocities for all joints.

    Args:
        dof_vel: DoF velocities tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_vel: Target DoF velocities tensor of shape (B, D).
    """
    
    required_keys = ("dof_vel", "target_dof_vel")

    def _compute(self, dof_vel: torch.Tensor, target_dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel[:, :] - target_dof_vel[:, :]), dim=1)

class TeleopDofVelSelected(RewardTerm):
    """
    Reward the similarity between the current and target DoF velocities for selected joints.

    Args:
        dof_vel: DoF velocities tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_vel: Target DoF velocities tensor of shape (B, D).
        dof_vel_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_vel", "target_dof_vel", "dof_vel_idxs")

    def _compute(self, dof_vel: torch.Tensor, target_dof_vel: torch.Tensor, dof_vel_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel[:, dof_vel_idxs] - target_dof_vel[:, dof_vel_idxs]), dim=1)

class TeleopDofVelLower(RewardTerm):
    """
    Reward the similarity between the current and target DoF velocities for lower joints.

    Args:
        dof_vel: DoF velocities tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_vel: Target DoF velocities tensor of shape (B, D).
        lower_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_vel", "target_dof_vel", "lower_idxs")

    def _compute(self, dof_vel: torch.Tensor, target_dof_vel: torch.Tensor, lower_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel[:, lower_idxs] - target_dof_vel[:, lower_idxs]), dim=1)

class TeleopDofVelUpper(RewardTerm):
    """
    Reward the similarity between the current and target DoF velocities for upper joints.

    Args:
        dof_vel: DoF velocities tensor of shape (B, D) where B is the batch size and D is the number of DoFs.
        target_dof_vel: Target DoF velocities tensor of shape (B, D).
        upper_idxs: Indices of the joints to consider for the reward.
    """
    
    required_keys = ("dof_vel", "target_dof_vel", "upper_idxs")

    def _compute(self, dof_vel: torch.Tensor, target_dof_vel: torch.Tensor, upper_idxs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return -torch.sum(torch.square(dof_vel[:, upper_idxs] - target_dof_vel[:, upper_idxs]), dim=1)