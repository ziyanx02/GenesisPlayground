"""Reward terms for manipulation tasks."""
import torch

from .reward_terms import RewardTerm


### ---- Main Task Rewards ---- ###


class CubeZRotationVelocityReward(RewardTerm):
    """
    Reward for cube rotating around Z-axis.
    Encourages positive angular velocity in the Z direction.

    Args:
        cube_ang_vel: Cube angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("cube_ang_vel",)

    def _compute(self, cube_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Reward Z-axis angular velocity (index 2)
        # Use absolute value to reward rotation in either direction
        return torch.abs(cube_ang_vel[:, 2])


class CubeZRotationVelocityDirectionalReward(RewardTerm):
    """
    Reward for cube rotating around Z-axis in a specific direction.
    Positive scale encourages counter-clockwise rotation (positive Z angular velocity).
    Negative scale encourages clockwise rotation (negative Z angular velocity).

    Args:
        cube_ang_vel: Cube angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("cube_ang_vel",)

    def _compute(self, cube_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Reward Z-axis angular velocity (index 2), directional
        return cube_ang_vel[:, 2]


class CubeStabilityPenalty(RewardTerm):
    """
    Penalize cube linear velocity to encourage stable manipulation.

    Args:
        cube_lin_vel: Cube linear velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("cube_lin_vel",)

    def _compute(self, cube_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Penalize linear motion
        return -torch.sum(cube_lin_vel**2, dim=-1)


class CubeXYRotationPenalty(RewardTerm):
    """
    Penalize cube rotation around X and Y axes to keep it upright.

    Args:
        cube_ang_vel: Cube angular velocity tensor of shape (B, 3) where B is the batch size.
    """

    required_keys = ("cube_ang_vel",)

    def _compute(self, cube_ang_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Penalize X and Y angular velocities to keep cube upright
        return -torch.sum(cube_ang_vel[:, :2] ** 2, dim=-1)


class CubeHeightPenalty(RewardTerm):
    """
    Penalize deviation from target cube height.

    Args:
        cube_pos: Cube position tensor of shape (B, 3) where B is the batch size.
        hand_palm_pos: Hand palm position tensor of shape (B, 3).
    """

    required_keys = ("cube_pos", "hand_palm_pos")

    def __init__(self, scale: float = 1.0, target_height_offset: float = 0.05, name: str | None = None):
        super().__init__(scale, name)
        self.target_height_offset = target_height_offset

    def _compute(self, cube_pos: torch.Tensor, hand_palm_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Penalize deviation from target height relative to hand
        height_diff = cube_pos[:, 2] - hand_palm_pos[:, 2] - self.target_height_offset
        return -torch.square(height_diff)


### ---- Regularization Penalties ---- ###


class ActionRatePenalty(RewardTerm):
    """
    Penalize large changes in actions between timesteps.

    Args:
        action_history_flat: Flattened action history of shape (B, D * H) where
                             D is action dimension and H is history length.
    """

    required_keys = ("action_history_flat",)

    def _compute(self, action_history_flat: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Reshape to (B, D, H)
        batch_size = action_history_flat.shape[0]
        # Assuming 20 DOF and at least 2 timesteps in history
        dof_dim = 20
        history_len = action_history_flat.shape[1] // dof_dim
        action_history = action_history_flat.reshape(batch_size, dof_dim, history_len)

        # Compute difference between last two actions
        if history_len >= 2:
            action_diff = action_history[:, :, -1] - action_history[:, :, -2]
            return -torch.sum(action_diff**2, dim=-1)
        else:
            return torch.zeros(batch_size, device=action_history_flat.device)


class ActionLimitPenalty(RewardTerm):
    """
    Penalize actions that are close to their limits.

    Args:
        action_history_flat: Flattened action history of shape (B, D * H).
    """

    required_keys = ("action_history_flat",)

    def __init__(self, scale: float = 1.0, limit: float = 0.9, name: str | None = None):
        super().__init__(scale, name)
        self.limit = limit

    def _compute(self, action_history_flat: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Get latest action
        batch_size = action_history_flat.shape[0]
        dof_dim = 20
        history_len = action_history_flat.shape[1] // dof_dim
        action_history = action_history_flat.reshape(batch_size, dof_dim, history_len)

        latest_action = action_history[:, :, -1]

        # Penalize actions beyond limit threshold
        over_limit = torch.abs(latest_action) > self.limit
        penalty = over_limit.float() * torch.abs(latest_action)
        return -torch.sum(penalty, dim=-1)


class DofPosLimitPenalty(RewardTerm):
    """
    Penalize DOF positions that are close to their limits.

    Args:
        hand_dof_pos: Hand DOF positions of shape (B, D).
        dof_pos_limits: DOF position limits of shape (D, 2).
    """

    required_keys = ("hand_dof_pos",)

    def __init__(self, scale: float = 1.0, margin: float = 0.1, name: str | None = None):
        super().__init__(scale, name)
        self.margin = margin

    def _compute(self, hand_dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Note: We would need dof_pos_limits from the robot, but for simplicity
        # we assume normalized positions and penalize values close to -1 or 1
        # This should be improved by passing actual limits from the environment

        # Penalize positions beyond 1 - margin
        over_limit_high = torch.relu(hand_dof_pos - (1.0 - self.margin))
        over_limit_low = torch.relu(-(hand_dof_pos + (1.0 - self.margin)))

        penalty = over_limit_high + over_limit_low
        return -torch.sum(penalty**2, dim=-1)


class DofVelLimitPenalty(RewardTerm):
    """
    Penalize high DOF velocities.

    Args:
        hand_dof_vel: Hand DOF velocities of shape (B, D).
    """

    required_keys = ("hand_dof_vel",)

    def __init__(self, scale: float = 1.0, threshold: float = 10.0, name: str | None = None):
        super().__init__(scale, name)
        self.threshold = threshold

    def _compute(self, hand_dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Penalize velocities beyond threshold
        over_threshold = torch.relu(torch.abs(hand_dof_vel) - self.threshold)
        return -torch.sum(over_threshold**2, dim=-1)


### ---- Contact-based Rewards ---- ###


class FingertipContactReward(RewardTerm):
    """
    Reward for maintaining contact with fingertips.
    (For future use if tactile sensors are added)

    Args:
        fingertip_contact_forces: Fingertip contact forces of shape (B, N, 3)
                                   where N is number of fingertips.
    """

    required_keys = ("fingertip_contact_forces",)

    def __init__(self, scale: float = 1.0, contact_threshold: float = 1.0, name: str | None = None):
        super().__init__(scale, name)
        self.contact_threshold = contact_threshold

    def _compute(self, fingertip_contact_forces: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute force magnitude for each fingertip
        force_magnitudes = torch.norm(fingertip_contact_forces, dim=-1)  # (B, N)

        # Count fingertips in contact
        in_contact = (force_magnitudes > self.contact_threshold).float()
        num_in_contact = torch.sum(in_contact, dim=-1)  # (B,)

        return num_in_contact


### ---- Energy Efficiency ---- ###


class EnergyEfficiencyPenalty(RewardTerm):
    """
    Penalize energy consumption (torque * velocity).

    Args:
        hand_dof_vel: Hand DOF velocities of shape (B, D).
        action_history_flat: Action history to approximate torque.
    """

    required_keys = ("hand_dof_vel", "action_history_flat")

    def _compute(self, hand_dof_vel: torch.Tensor, action_history_flat: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Get latest action as proxy for torque
        batch_size = action_history_flat.shape[0]
        dof_dim = 20
        history_len = action_history_flat.shape[1] // dof_dim
        action_history = action_history_flat.reshape(batch_size, dof_dim, history_len)
        latest_action = action_history[:, :, -1]

        # Power = torque * velocity (approximated by action * velocity)
        power = torch.abs(latest_action * hand_dof_vel)
        return -torch.sum(power, dim=-1)


### ---- Penspin-style Reward Terms ---- ###


class RotateRewardClipped(RewardTerm):
    """
    Reward for rotation around a specific axis with clipping.
    Similar to penspin's rotate_reward calculation.

    Args:
        cube_ang_vel: Cube angular velocity tensor of shape (B, 3).
        rot_axis: Rotation axis tensor of shape (B, 3) indicating the target rotation axis.
    """

    required_keys = ("cube_ang_vel", "rot_axis")

    def __init__(
        self,
        scale: float = 1.0,
        angvel_scale_factor: float = 1.0,
        angvel_clip_min: float = -4.0,
        angvel_clip_max: float = 4.0,
        name: str | None = None,
    ):
        super().__init__(scale, name)
        self.angvel_scale_factor = angvel_scale_factor
        self.angvel_clip_min = angvel_clip_min
        self.angvel_clip_max = angvel_clip_max

    def _compute(self, cube_ang_vel: torch.Tensor, rot_axis: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Compute dot product between angular velocity and rotation axis
        vec_dot = torch.sum(cube_ang_vel * rot_axis, dim=-1)
        # Clip the reward
        rotate_reward = torch.clamp(vec_dot * self.angvel_scale_factor, min=self.angvel_clip_min, max=self.angvel_clip_max)
        return rotate_reward


class RotatePenaltyThreshold(RewardTerm):
    """
    Penalty for excessive rotation velocity beyond a threshold.
    Similar to penspin's rotate_penalty calculation.

    Args:
        cube_ang_vel: Cube angular velocity tensor of shape (B, 3).
        rot_axis: Rotation axis tensor of shape (B, 3).
    """

    required_keys = ("cube_ang_vel", "rot_axis")

    def __init__(
        self,
        scale: float = 1.0,
        angvel_penalty_threshold: float = 3.0,
        name: str | None = None,
    ):
        super().__init__(scale, name)
        self.angvel_penalty_threshold = angvel_penalty_threshold

    def _compute(self, cube_ang_vel: torch.Tensor, rot_axis: torch.Tensor) -> torch.Tensor:  # type: ignore
        vec_dot = torch.sum(cube_ang_vel * rot_axis, dim=-1)
        # Penalize only when exceeding threshold
        rotate_penalty = torch.where(
            vec_dot > self.angvel_penalty_threshold,
            vec_dot - self.angvel_penalty_threshold,
            torch.zeros_like(vec_dot),
        )
        return -rotate_penalty


class ObjectLinVelPenalty(RewardTerm):
    """
    Penalty for object linear velocity (L1 norm).
    Similar to penspin's object_linvel_penalty.

    Args:
        cube_lin_vel: Cube linear velocity tensor of shape (B, 3).
    """

    required_keys = ("cube_lin_vel",)

    def _compute(self, cube_lin_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        # L1 norm of linear velocity
        object_linvel_penalty = torch.norm(cube_lin_vel, p=1, dim=-1)
        return -object_linvel_penalty


class PoseDiffPenalty(RewardTerm):
    """
    Penalty for deviation from initial pose.
    Similar to penspin's pose_diff_penalty.

    Args:
        hand_dof_pos: Current hand DOF positions of shape (B, D).
        init_dof_pos: Initial hand DOF positions of shape (B, D).
    """

    required_keys = ("hand_dof_pos", "init_dof_pos")

    def _compute(self, hand_dof_pos: torch.Tensor, init_dof_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        pose_diff = hand_dof_pos - init_dof_pos
        pose_diff_penalty = torch.sum(pose_diff**2, dim=-1)
        return -pose_diff_penalty


class TorquePenalty(RewardTerm):
    """
    Penalty for torque squared.
    Similar to penspin's torque_penalty.

    Args:
        torques: Torque values of shape (B, D).
    """

    required_keys = ("torques",)

    def _compute(self, torques: torch.Tensor) -> torch.Tensor:  # type: ignore
        torque_penalty = torch.sum(torques**2, dim=-1)
        return -torque_penalty


class WorkPenalty(RewardTerm):
    """
    Penalty for mechanical work done (sum of |torque × velocity|)^2.
    Similar to penspin's work_penalty.

    Args:
        torques: Torque values of shape (B, D).
        hand_dof_vel: Hand DOF velocities of shape (B, D).
    """

    required_keys = ("torques", "hand_dof_vel")

    def _compute(self, torques: torch.Tensor, hand_dof_vel: torch.Tensor) -> torch.Tensor:  # type: ignore
        work = torch.sum(torch.abs(torques) * torch.abs(hand_dof_vel), dim=-1)
        work_penalty = work**2
        return -work_penalty


class PositionPenalty(RewardTerm):
    """
    Penalty for object position deviation from target.
    Similar to penspin's position_penalty.

    Args:
        cube_pos: Cube position tensor of shape (B, 3).
        target_pos: Target position tensor of shape (B, 3) or (3,).
    """

    required_keys = ("cube_pos",)

    def __init__(self, scale: float = 1.0, target_x: float = 0.0, target_y: float = 0.0, target_z: float = 0.61, name: str | None = None):
        super().__init__(scale, name)
        self.target_x = target_x
        self.target_y = target_y
        self.target_z = target_z

    def _compute(self, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        position_penalty = (
            (cube_pos[:, 0] - self.target_x) ** 2
            + (cube_pos[:, 1] - self.target_y) ** 2
            + (cube_pos[:, 2] - self.target_z) ** 2
        )
        return -position_penalty


class FingerObjectDistancePenalty(RewardTerm):
    """
    Penalty for distance between fingertips and object.
    Similar to penspin's finger_obj_penalty.

    Args:
        fingertip_pos: Fingertip positions of shape (B, N*3) where N is number of fingertips.
        cube_pos: Cube position tensor of shape (B, 3).
    """

    required_keys = ("fingertip_pos", "cube_pos")

    def _compute(self, fingertip_pos: torch.Tensor, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Auto-detect number of fingertips from shape
        num_fingertips = fingertip_pos.shape[1] // 3
        # Repeat cube position for each fingertip
        cube_pos_repeated = cube_pos.repeat(1, num_fingertips)
        # Compute squared distance
        finger_obj_penalty = torch.sum((fingertip_pos - cube_pos_repeated) ** 2, dim=-1)
        return -finger_obj_penalty


class FingertipCubeProximityPenalty(RewardTerm):
    """
    Penalty for fingertips being far from the cube.
    Computes the average distance from all fingertips to the cube center.

    This encourages the hand to maintain contact/proximity with the cube during manipulation.

    Args:
        fingertip_pos: Fingertip positions of shape (B, N*3) where N is number of fingertips (flattened).
        cube_pos: Cube position tensor of shape (B, 3).
    """

    required_keys = ("fingertip_pos", "cube_pos")

    def __init__(self, scale: float = 1.0, distance_threshold: float | None = None, name: str | None = None):
        """
        Args:
            scale: Reward scale factor.
            distance_threshold: Optional threshold distance. If specified, only penalize distances beyond this threshold.
        """
        super().__init__(scale, name)
        self.distance_threshold = distance_threshold

    def _compute(self, fingertip_pos: torch.Tensor, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch_size = fingertip_pos.shape[0]
        num_fingertips = fingertip_pos.shape[1] // 3

        # Reshape fingertip_pos to (B, N, 3)
        fingertip_pos_reshaped = fingertip_pos.reshape(batch_size, num_fingertips, 3)

        # Expand cube_pos to (B, 1, 3) for broadcasting
        cube_pos_expanded = cube_pos.unsqueeze(1)

        # Compute distance from each fingertip to cube center: (B, N)
        distances = torch.norm(fingertip_pos_reshaped - cube_pos_expanded, dim=-1)

        # Apply threshold if specified
        if self.distance_threshold is not None:
            # Only penalize distances beyond threshold
            distances = torch.relu(distances - self.distance_threshold)

        # Average distance across all fingertips
        avg_distance = torch.mean(distances, dim=-1)

        # Return negative penalty (closer = less penalty)
        return -avg_distance


class FingertipCubeProximityPenaltySquared(RewardTerm):
    """
    Penalty for fingertips being far from the cube (squared distance version).
    Uses squared distances which penalizes larger deviations more heavily.

    This encourages the hand to maintain contact/proximity with the cube during manipulation.

    Args:
        fingertip_pos: Fingertip positions of shape (B, N*3) where N is number of fingertips (flattened).
        cube_pos: Cube position tensor of shape (B, 3).
    """

    required_keys = ("fingertip_pos", "cube_pos")

    def _compute(self, fingertip_pos: torch.Tensor, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        batch_size = fingertip_pos.shape[0]
        num_fingertips = fingertip_pos.shape[1] // 3

        # Reshape fingertip_pos to (B, N, 3)
        fingertip_pos_reshaped = fingertip_pos.reshape(batch_size, num_fingertips, 3)

        # Expand cube_pos to (B, 1, 3) for broadcasting
        cube_pos_expanded = cube_pos.unsqueeze(1)

        # Compute squared distance from each fingertip to cube center: (B, N)
        squared_distances = torch.sum((fingertip_pos_reshaped - cube_pos_expanded) ** 2, dim=-1)

        # Average squared distance across all fingertips
        avg_squared_distance = torch.mean(squared_distances, dim=-1)

        # Return negative penalty (closer = less penalty)
        return -avg_squared_distance

class FingertipCubeProximityReward(RewardTerm):
      """
      Reward for fingertips being close to the cube.
      Encourages maintaining grasp.
      """
      required_keys = ("fingertip_pos", "cube_pos")

      def __init__(self, scale: float = 1.0, threshold: float = 0.05, name: str | None = None):
          super().__init__(scale, name)
          self.threshold = threshold

      def _compute(self, fingertip_pos: torch.Tensor, cube_pos: torch.Tensor) -> torch.Tensor:
          batch_size = fingertip_pos.shape[0]
          num_fingertips = fingertip_pos.shape[1] // 3

          fingertip_pos_reshaped = fingertip_pos.reshape(batch_size, num_fingertips, 3)
          cube_pos_expanded = cube_pos.unsqueeze(1)

          # Distance from each fingertip to cube
          distances = torch.norm(fingertip_pos_reshaped - cube_pos_expanded, dim=-1)

          # Reward fingertips within threshold
          in_contact = (distances < self.threshold).float()
          num_in_contact = in_contact.sum(dim=-1)

          return num_in_contact  # Reward proportional to number of fingertips in contact


class CubeOnHandReward(RewardTerm):
    """
    Reward for keeping the cube on the hand (not dropped).
    Provides a constant positive reward when the cube is above the hand
    and within XY distance threshold.

    This encourages the policy to maintain grasp and avoid dropping the cube,
    which should increase episode length.

    Args:
        cube_pos: Cube position tensor of shape (B, 3).
        hand_palm_pos: Hand palm position tensor of shape (B, 3).
    """

    required_keys = ("cube_pos",)

    def __init__(
        self,
        scale: float = 1.0,
        stay_center: list[float] = [0.0, 0.0, 0.15],
        height_threshold: float = -0.05,  # Match termination threshold
        xy_threshold: float = 0.15,  # Match termination threshold
        name: str | None = None
    ):
        """
        Args:
            scale: Reward scale factor.
            height_threshold: Minimum height above hand (negative = below hand palm).
            xy_threshold: Maximum XY distance from hand center.
        """
        super().__init__(scale, name)
        self.stay_center = torch.tensor(stay_center).unsqueeze(0)  # (1, 3)
        self.height_threshold = height_threshold
        self.xy_threshold = xy_threshold

    def _compute(self, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Check height constraint (cube should be above threshold)
        cube_height_above_hand = cube_pos[:, 2] - self.stay_center[:, 2]
        height_ok = (cube_height_above_hand >= self.height_threshold).float()

        # Check XY distance constraint (cube should be near hand center)
        cube_xy_dist = torch.norm(cube_pos[:, :2] - self.stay_center[:, :2], dim=-1)
        xy_ok = (cube_xy_dist <= self.xy_threshold).float()

        # Both constraints must be satisfied
        on_hand = height_ok * xy_ok

        return on_hand  # Returns 1.0 when on hand, 0.0 when dropped


class EarlyTerminationPenalty(RewardTerm):
    """
    Penalty applied when episode terminates early (cube dropped or too far).
    Provides a one-time large negative reward at termination to discourage dropping.

    This is computed based on whether the cube violates termination conditions,
    giving a strong signal to avoid those states.

    Args:
        cube_pos: Cube position tensor of shape (B, 3).
    """

    required_keys = ("cube_pos",)

    def __init__(
        self,
        scale: float = 1.0,
        stay_center: list[float] = [0.0, 0.0, 0.15],
        height_threshold: float = -0.05,
        xy_threshold: float = 0.15,
        name: str | None = None
    ):
        """
        Args:
            scale: Penalty scale factor (use negative value for penalty).
            stay_center: Center position to compare against (typically hand position).
            height_threshold: Minimum height above center.
            xy_threshold: Maximum XY distance from center.
        """
        super().__init__(scale, name)
        self.stay_center = stay_center
        self.height_threshold = height_threshold
        self.xy_threshold = xy_threshold

    def _compute(self, cube_pos: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Convert stay_center to tensor on the same device as cube_pos
        stay_center_tensor = torch.tensor(self.stay_center, device=cube_pos.device).unsqueeze(0)

        # Check if cube is dropped (height below threshold)
        cube_height_above_center = cube_pos[:, 2] - stay_center_tensor[:, 2]
        is_dropped = (cube_height_above_center < self.height_threshold).float()

        # Check if cube is too far in XY
        cube_xy_dist = torch.norm(cube_pos[:, :2] - stay_center_tensor[:, :2], dim=-1)
        is_too_far = (cube_xy_dist > self.xy_threshold).float()

        # Penalty if either condition is violated (will be multiplied by negative scale)
        terminated = torch.clamp(is_dropped + is_too_far, 0.0, 1.0)

        return -terminated  # Returns -1.0 when terminated, 0.0 when safe