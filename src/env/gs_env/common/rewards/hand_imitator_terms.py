"""Reward terms for hand trajectory imitation tasks.

Following ManipTrans implementation from:
maniptrans_envs/lib/envs/tasks/dexhandimitator.py (lines 1065-1201)
"""
import torch

from gs_env.common.utils.math_utils import quat_conjugate, quat_mul

from .reward_terms import RewardTerm


### ---- Trajectory Following Rewards ---- ###


class WristPositionTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist position.
    Uses exponential reward: exp(-k * position_error)

    Required state keys:
        base_pos: Current wrist position (B, 3)
        target_wrist_pos: Target wrist position from trajectory (B, K * 3)
    """

    required_keys = ("base_pos", "target_wrist_pos")

    def __init__(self, scale: float = 0.1, k: float = 40.0, name: str | None = None):
        """
        Args:
            scale: Overall reward scale
            k: Exponential decay rate (higher = sharper reward)
        """
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_pos: torch.Tensor, target_wrist_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_wrist_pos[..., :3] - base_pos
        dist = torch.norm(diff, dim=-1)
        return torch.exp(-self.k * dist)


class ObjectPositionTrackingReward(RewardTerm):
    """
    Reward for tracking target object position.
    Uses exponential reward: exp(-k * position_error)

    Required state keys:
        object_pos: Current object position (B, 3)
        target_object_pos: Target object position from trajectory (B, K, 3)
    """

    required_keys = ("object_pos", "target_object_pos")

    def __init__(self, scale: float = 5.0, k: float = 80.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
    
    def _compute(self, object_pos: torch.Tensor, target_object_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_object_pos[:, 0] - object_pos
        dist = torch.norm(diff, dim=-1)
        return torch.exp(-self.k * dist)


class WristRotationTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist rotation.
    Uses exponential reward based on rotation angle difference.

    Required state keys:
        base_quat: Current wrist quaternion (B, 4) [w, x, y, z]
        target_wrist_quat: Target wrist quaternion (B, K * 4) [w, x, y, z]
    """

    required_keys = ("base_quat", "target_wrist_quat")

    def __init__(self, scale: float = 0.6, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_quat: torch.Tensor, target_wrist_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute delta quaternion: q_delta = q_target * q_current^{-1}
        # Note: Using standard quaternion multiplication
        diff_quat = quat_mul(target_wrist_quat[:, :4], quat_conjugate(base_quat))

        # Extract rotation angle from quaternion
        # angle = 2 * acos(w), clamped to [-1, 1]
        w = diff_quat[..., 0]
        angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

        return torch.exp(-self.k * torch.abs(angle))


class ObjectRotationTrackingReward(RewardTerm):

    required_keys = ("object_quat", "target_object_quat")

    def __init__(self, scale: float = 1.0, k: float = 3.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_quat: torch.Tensor, target_object_quat: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff_quat = quat_mul(target_object_quat[:, :4], quat_conjugate(object_quat))
        w = diff_quat[..., 0]
        angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))

        return torch.exp(-self.k * torch.abs(angle))


class WristVelocityTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist linear velocity.

    Required state keys:
        base_lin_vel: Current wrist linear velocity (B, 3)
        target_wrist_vel: Target wrist linear velocity (B, K * 3)
    """

    required_keys = ("base_lin_vel", "target_wrist_vel")

    def __init__(self, scale: float = 0.1, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_lin_vel: torch.Tensor, target_wrist_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_wrist_vel[..., :3] - base_lin_vel
        error = torch.abs(diff).mean(dim=-1)
        return torch.exp(-self.k * error)


class ObjectVelocityTrackingReward(RewardTerm):

    required_keys = ("object_lin_vel", "target_object_vel")

    def __init__(self, scale: float = 0.1, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_lin_vel: torch.Tensor, target_object_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_object_vel[:, :3] - object_lin_vel
        error = torch.abs(diff).mean(dim=-1)
        return torch.exp(-self.k * error)


class WristAngularVelocityTrackingReward(RewardTerm):
    """
    Reward for tracking target wrist angular velocity.

    Required state keys:
        base_ang_vel: Current wrist angular velocity (B, 3)
        target_wrist_ang_vel: Target wrist angular velocity (B, K * 3)
    """

    required_keys = ("base_ang_vel", "target_wrist_ang_vel")

    def __init__(self, scale: float = 0.05, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, base_ang_vel: torch.Tensor, target_wrist_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_wrist_ang_vel[..., :3] - base_ang_vel
        error = torch.abs(diff).mean(dim=-1)
        return torch.exp(-self.k * error)
    

class ObjectAngularVelocityTrackingReward(RewardTerm):

    required_keys = ("object_ang_vel", "target_object_ang_vel")

    def __init__(self, scale: float = 0.1, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, object_ang_vel: torch.Tensor, target_object_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        diff = target_object_ang_vel[:, :3] - object_ang_vel
        error = torch.abs(diff).mean(dim=-1)
        return torch.exp(-self.k * error)


class FingerJointPositionTrackingReward(RewardTerm):
    """
    Reward for tracking target finger joint positions in 3D space.
    Supports per-finger weighting (thumb, index, middle, ring, pinky).

    Required state keys:
        finger_link_pos: Current finger link positions (B, 20, 3)
        target_mano_joint_pos: Target joint positions (B, K, 20, 3)
    """

    required_keys = ("finger_link_pos", "target_mano_joint_pos")

    def __init__(
        self,
        scale: float = 1.0,
        thumb_weight: float = 0.9,
        index_weight: float = 0.8,
        middle_weight: float = 0.75,
        ring_weight: float = 0.6,
        pinky_weight: float = 0.6,
        level_1_weight: float = 0.5,
        level_2_weight: float = 0.3,
        thumb_k: float = 100.0,
        index_k: float = 90.0,
        middle_k: float = 80.0,
        ring_k: float = 60.0,
        pinky_k: float = 60.0,
        level_1_k: float = 50.0,
        level_2_k: float = 40.0,
        name: str | None = None,
    ):
        """
        Args:
            scale: Overall reward scale
            *_weight: Per-finger weights for final reward combination
            *_k: Exponential decay rates for each finger type
        """
        super().__init__(scale, name)
        self.thumb_weight = thumb_weight
        self.index_weight = index_weight
        self.middle_weight = middle_weight
        self.ring_weight = ring_weight
        self.pinky_weight = pinky_weight
        self.level_1_weight = level_1_weight
        self.level_2_weight = level_2_weight
        self.thumb_k = thumb_k
        self.index_k = index_k
        self.middle_k = middle_k
        self.ring_k = ring_k
        self.pinky_k = pinky_k
        self.level_1_k = level_1_k
        self.level_2_k = level_2_k

    def _compute(self, finger_link_pos: torch.Tensor, target_mano_joint_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute position differences
        diff_joints_pos = target_mano_joint_pos[:, 0] - finger_link_pos  # (B, 20, 3)
        diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)  # (B, 20)

        diff_thumb_tip = diff_joints_pos_dist[:, 3]
        diff_index_tip = diff_joints_pos_dist[:, 7]
        diff_middle_tip = diff_joints_pos_dist[:, 11]
        diff_ring_tip = diff_joints_pos_dist[:, 15]
        diff_pinky_tip = diff_joints_pos_dist[:, 19]
        diff_level_1 = diff_joints_pos_dist[:, [0, 4, 8, 12, 16]].mean(dim=-1)
        diff_level_2 = diff_joints_pos_dist[:, [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]].mean(dim=-1)

        # Compute per-finger rewards
        reward_thumb = torch.exp(-self.thumb_k * diff_thumb_tip)
        reward_index = torch.exp(-self.index_k * diff_index_tip)
        reward_middle = torch.exp(-self.middle_k * diff_middle_tip)
        reward_ring = torch.exp(-self.ring_k * diff_ring_tip)
        reward_pinky = torch.exp(-self.pinky_k * diff_pinky_tip)
        reward_level_1 = torch.exp(-self.level_1_k * diff_level_1)
        reward_level_2 = torch.exp(-self.level_2_k * diff_level_2)
        # Combine with weights
        total_reward = (
            self.thumb_weight * reward_thumb
            + self.index_weight * reward_index
            + self.middle_weight * reward_middle
            + self.ring_weight * reward_ring
            + self.pinky_weight * reward_pinky
            + self.level_1_weight * reward_level_1
            + self.level_2_weight * reward_level_2
        )

        return total_reward


class JointVelocityTrackingReward(RewardTerm):
    """
    Reward for tracking target joint velocities.

    Required state keys:
        finger_link_vel: Current finger link velocities (B, 20, 3)
        target_mano_joint_vel: Target mano joint velocities (B, K * 20 * 3)
    """

    required_keys = ("finger_link_vel", "target_mano_joint_vel")

    def __init__(self, scale: float = 0.1, k: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, finger_link_vel: torch.Tensor, target_mano_joint_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute velocity differences
        num_envs = finger_link_vel.shape[0]
        diff_joints_vel = target_mano_joint_vel[:, :60].reshape(num_envs, 20, 3) - finger_link_vel
        error = torch.abs(diff_joints_vel).mean(dim=-1).mean(dim=-1)  # Mean over joints and dimensions

        return torch.exp(-self.k * error)


### ---- Power Penalties ---- ###


class DOFPowerPenalty(RewardTerm):
    """
    Penalize actuator power consumption (torque * velocity).

    Required state keys:
        dof_force: DOF forces/torques (B, 6 + n_hand_dofs)
        hand_dof_vel: DOF velocities (B, n_hand_dofs)
    """

    required_keys = ("dof_force", "hand_dof_vel")

    def __init__(self, scale: float = 0.5, k: float = 10.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(self, dof_force: torch.Tensor, hand_dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Power = |torque * velocity|
        power = torch.abs(dof_force[:, 6:] * hand_dof_vel).sum(dim=-1)
        return torch.exp(-self.k * power)


class WristPowerPenalty(RewardTerm):
    """
    Penalize wrist control power (force * velocity + torque * angular velocity).

    Required state keys:
        dof_force: DOF forces/torques (B, 6 + n_hand_dofs)
        base_lin_vel: Wrist linear velocity (B, 3)
        base_ang_vel: Wrist angular velocity (B, 3)
    """

    required_keys = ("dof_force", "base_lin_vel", "base_ang_vel")

    def __init__(self, scale: float = 0.5, k: float = 2.0, name: str | None = None):
        super().__init__(scale, name)
        self.k = k

    def _compute(  # type: ignore
        self,
        dof_force: torch.Tensor,
        base_lin_vel: torch.Tensor,
        base_ang_vel: torch.Tensor,
    ) -> torch.Tensor:
        # Linear power
        linear_power = torch.abs(torch.sum(dof_force[:, :3] * base_lin_vel, dim=-1))

        # Angular power
        angular_power = torch.abs(torch.sum(dof_force[:, 3:6] * base_ang_vel, dim=-1))

        total_power = linear_power + angular_power
        return torch.exp(-self.k * total_power)


class FingertipForceReward(RewardTerm):

    required_keys = ("fingertip_max_force", "mano_fingertip_to_object")

    def __init__(self, scale: float = 1.0, k: float = 1.0, range_min: float = 0.02, range_max: float = 0.03, name: str | None = None):
        super().__init__(scale, name)
        self.k = k
        self.range_min = range_min
        self.range_max = range_max

    def _compute(self, fingertip_max_force: torch.Tensor, mano_fingertip_to_object: torch.Tensor) -> torch.Tensor:  # type: ignore
        # fingertip_max_force: (B, 5), mano_fingertip_to_object: (B, K * 5)
        dist_to_object = mano_fingertip_to_object[:, :5]  # (B, 5)
        
        fingertip_weight = torch.clamp(
            (self.range_max - dist_to_object) / (self.range_max - self.range_min),
            min=0.0, max=1.0
        )  # (B, 5)

        force_masked = fingertip_max_force * fingertip_weight  # (B, 5)
        force_sum = force_masked.sum(dim=-1)  # (B,)

        return torch.exp(-self.k * (1 / (force_sum + 1e-5)))  # Avoid division by zero