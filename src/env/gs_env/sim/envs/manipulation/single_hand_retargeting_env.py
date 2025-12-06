"""Hand trajectory following environment for WUJI hand.

This environment trains a policy to follow reference hand trajectories from demonstrations.
Similar to ManipTrans DexHandImitator but implemented in Genesis.
"""
import importlib
import pickle
from pathlib import Path
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
import trimesh
from bps_torch.bps import bps_torch
from PIL import Image

from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    quat_to_angle_axis,
    quat_to_euler,
    quat_to_rotmat,
    rotmat_to_quat,
)
from scipy.spatial.transform import Rotation as R
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.common.utils.maniptrans_util import (
    compute_velocity,
    compute_angular_velocity,
    compute_dof_velocity,
    axis_angle_to_rotmat_batch,
    rotmat_to_axis_angle_batch,
)
from gs_env.sim.envs.config.schema import SingleHandRetargetingEnvArgs
from gs_env.sim.envs.manipulation.tactile_visualizer import TactileVisualizer
from gs_env.sim.robots.manipulators import WUJIHand
from gs_env.sim.scenes import FlatScene
from gs_agent.algos.config.registry import PPO_HAND_IMITATOR_MLP
from gs_agent.modules.models import NetworkFactory
from gs_agent.modules.policies import GaussianPolicy

_DEFAULT_DEVICE = torch.device("cpu")


# TODO:
# 1, add mano reference finger tip to object distance
# 2, add tactile sonsor
# 3, add base model for residual learning


class SingleHandRetargetingEnv(BaseEnv):
    """
    Hand trajectory following environment for WUJI dexterous hand with object manipulation.
    Task: Follow reference hand trajectories from demonstration data while manipulating objects.

    Observations (three components following ManipTrans architecture):
    - Proprioception: hand_dof_pos (cos/sin encoded), base_state (pos, quat, vel, ang_vel)
    - Privileged: hand_dof_vel, object state
    - Target: reference trajectory deltas and targets for future timesteps (both hand and object)

    Actions (20D + 6D base): Delta joint positions + base pose control

    Rewards:
        - Hand trajectory following rewards (position, rotation, velocity matching)
        - Object trajectory following rewards (position, rotation, velocity matching)
        - Power penalties
    """

    def __init__(
        self,
        args: SingleHandRetargetingEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._args = args
        self._eval_mode = eval_mode

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))

        # == Load demonstration trajectory data ==
        self._load_trajectory_data()

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
        )

        # == setup the robot (WUJI hand) ==
        # For trajectory following, the hand should be free-floating (is_free=True)
        self._robot = WUJIHand(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self.device,
        )
        self._hand_dof_dim = self._robot._gripper_dof_dim
        self._base_dof_dim = self._robot._arm_dof_dim

        # == setup object ==
        obj_id = self._args.object_id
        self._object = self._scene.scene.add_entity(
            gs.morphs.Mesh(
                file=f"{self._args.trajectory_path}/{obj_id}_collision.obj",
                pos=[0.0, 0.0, 0.0],  # random init pose
                euler=(90, 0, 0),
            ),
            material=gs.materials.Rigid(
                rho=100.0,
                friction=0.8,
            ),
        )

        # == set up camera for rendering ==
        self._floating_camera = self._scene.scene.add_camera(
            res=(480, 480),
            fov=60,
            GUI=False,
        )

        # == setup tactile sensors ==
        self._use_tactile = args.use_tactile
        self._tactile_sensors = {}
        self._tactile_points = {}
        self._tactile_sensor_configs = {}
        self._tactile_visualizer = None

        if self._use_tactile:
            assert "tactile_forces_flat" in args.actor_obs_terms and "tactile_forces_flat" in args.critic_obs_terms, \
                "tactile_forces_flat must be included in both actor and critic observation terms when use_tactile is True"
            self._setup_tactile_sensors(args)

        # == build the scene ==
        self._scene.build()

        # == initialize robot limits after scene is built ==
        self._robot.post_build_init(eval_mode=eval_mode)

        # == setup tactile visualization if enabled ==
        if self._use_tactile and show_viewer and eval_mode and False:
            print(f"\n{'='*70}")
            print(f"Initializing tactile visualization...")
            print(f"{'='*70}")
            self._tactile_visualizer = TactileVisualizer(
                tactile_sensors=self._tactile_sensors,
                tactile_points=self._tactile_points,
                tactile_sensor_configs=self._tactile_sensor_configs,
            )
            print(f"✓ Tactile visualization ready")

        # == Process trajectory data for RL training ==
        self._process_trajectory_data()

        # == setup reward functions ==
        dt = self._scene.scene.dt
        self._reward_functions = {}
        self._reward_required_keys: set[str] = set()

        # Load reward terms from config
        reward_term = getattr(args, 'reward_term', 'hand_imitator')
        module_name = f"gs_env.common.rewards.{reward_term}_terms"
        module = importlib.import_module(module_name)

        reward_args = getattr(args, 'reward_args', {})
        for key in reward_args.keys():
            reward_cls = getattr(module, key, None)
            if reward_cls is None:
                raise ValueError(f"Reward {key} not found in rewards module {module_name}.")
            scale = reward_args[key]["scale"] * dt
            other_args = {k: v for k, v in reward_args[key].items() if k != "scale"}
            reward_instance = reward_cls(scale=scale, **other_args)
            self._reward_functions[key] = reward_instance
            # Record declared inputs for this reward term
            if hasattr(reward_instance, "required_keys"):
                self._reward_required_keys.update(getattr(reward_instance, "required_keys", ()))

        # Environment parameters
        self._max_episode_length = args.max_episode_length
        self._obs_future_length = args.obs_future_length

        # Initialize buffers
        self._init()

        # ===== Load base policy for residual learning if enabled =====
        self._use_residual_learning = args.use_residual_learning
        self._base_policy = None
        self._base_policy_obs_terms = None

        if self._use_residual_learning:
            if args.base_policy_path is None:
                raise ValueError("base_policy_path must be provided when use_residual_learning=True")
            if args.base_policy_obs_terms is None:
                raise ValueError("base_policy_obs_terms must be provided when use_residual_learning=True")

            print(f"\n{'='*80}")
            print(f"Loading base policy for residual learning from: {args.base_policy_path}")
            print(f"{'='*80}\n")

            # Store base policy observation terms
            self._base_policy_obs_terms = args.base_policy_obs_terms

            # Compute base policy observation dimension
            self._base_policy_obs_dim = self._compute_base_policy_obs_dim()
            self._base_action_dim = self._hand_dof_dim + self._base_dof_dim  # 26 for free base

            print(f"Base policy observation dimension: {self._base_policy_obs_dim}")
            print(f"Base policy action dimension: {self._base_action_dim}")
            
            # Create policy backbone
            policy_backbone = NetworkFactory.create_network(
                network_backbone_args=PPO_HAND_IMITATOR_MLP.policy_backbone,
                input_dim=self._base_policy_obs_dim,
                output_dim=self._base_action_dim,
                device=self._device,
            )

            # Create base policy
            self._base_policy = GaussianPolicy(
                policy_backbone=policy_backbone,
                action_dim=self._base_action_dim,
            ).to(self._device)

            # Load checkpoint
            checkpoint = torch.load(args.base_policy_path, map_location=self._device)
            self._base_policy.load_state_dict(checkpoint['model_state_dict'])

            # Set to eval mode
            self._base_policy.eval()

            print(f"Base policy loaded successfully!")
            print(f"Base policy checkpoint iteration: {checkpoint['iter']}")
            print(f"{'='*80}\n")

        self.reset()

    def _load_trajectory_data(self) -> None:
        """Load multiple demonstration trajectory data from pickle files."""
        trajectory_path = Path(self._args.trajectory_path)

        # Load all .pkl files from directory
        traj_files = sorted(trajectory_path.glob("*.pkl"))
        if not traj_files:
            raise FileNotFoundError(f"No trajectory files found in directory: {trajectory_path}")

        print(f"\n{'='*80}")
        print(f"Loading {len(traj_files)} trajectories from: {trajectory_path}")
        print(f"{'='*80}")

        all_demo_data = []
        all_wrist_positions = []
        all_wrist_rotations_aa = []
        all_dof_positions = []
        all_obj_pose_matrices = []
        all_traj_lengths = []

        for traj_file in traj_files:
            print(f"Loading: {traj_file.name}")
            with open(traj_file, "rb") as f:
                data = pickle.load(f)
            
            if data["metadata"]["obj_id"] != self._args.object_id:
                continue  # skip trajectories not matching the specified object_id

            if self._args.trajectory_id is not None:
                if self._args.trajectory_id not in traj_file.name:
                    continue  # skip if trajectory_id is specified and does not match

            all_demo_data.append(data)
            hand_traj = data["hand_trajectory"]

            wrist_pos = hand_traj["wrist_positions"]  # (T, 3)
            wrist_rot = hand_traj["wrist_rotations_aa"]  # (T, 3)
            dof_pos = hand_traj["dof_positions"]  # (T, n_dofs)
            obj_pose_matrix = data["object_trajectory"]["pose_matrices"]  # (T, 4, 4)

            all_wrist_positions.append(wrist_pos)
            all_wrist_rotations_aa.append(wrist_rot)
            all_dof_positions.append(dof_pos)
            all_obj_pose_matrices.append(obj_pose_matrix)
            all_traj_lengths.append(len(wrist_pos))

            print(f"  - Length: {len(wrist_pos)} timesteps")

            if len(all_demo_data) >= self._args.max_num_trajectories:
                break  # limit to max_num_trajectories
        
        if len(all_demo_data) == 0:
            raise ValueError(f"No trajectories found for object_id: {self._args.object_id}")

        # Store raw data and individual trajectory lengths
        self._demo_data_raw = all_demo_data
        self._num_trajectories = len(all_demo_data)
        self._traj_lengths = np.array(all_traj_lengths)
        self._max_traj_length = max(all_traj_lengths)

        # Pad and combine trajectories
        self._wrist_positions = self._pad_and_stack(all_wrist_positions)  # (num_traj, max_T, 3)
        self._wrist_rotations_aa = self._pad_and_stack(all_wrist_rotations_aa)  # (num_traj, max_T, 3)
        self._dof_positions = self._pad_and_stack(all_dof_positions)  # (num_traj, max_T, n_dofs)
        self._obj_pose_matrices = self._pad_and_stack(all_obj_pose_matrices)  # (num_traj, max_T, 4, 4)

        print(f"\nCombined trajectory shapes:")
        print(f"  - Wrist positions: {self._wrist_positions.shape}")
        print(f"  - Wrist rotations: {self._wrist_rotations_aa.shape}")
        print(f"  - DOF positions: {self._dof_positions.shape}")
        print(f"  - Max trajectory length: {self._max_traj_length}")
        print(f"  - Trajectory lengths: {self._traj_lengths}")
        print(f"{'='*80}\n")

    def _compute_base_policy_obs_dim(self) -> int:
        """Compute base policy observation dimension from observation terms.

        This should be called after _init() so all buffers are created.
        """
        total_dim = 0
        for key in self._args.base_policy_obs_terms:
            obs_buffer = getattr(self, key)
            if obs_buffer.dim() == 1:
                # Single environment buffer, dimension is the tensor size
                total_dim += obs_buffer.shape[0]
            else:
                # Multiple environment buffer (num_envs, dim)
                total_dim += obs_buffer.shape[1]
        return total_dim

    def _pad_and_stack(self, arrays: list[np.ndarray]) -> np.ndarray:
        """Pad arrays to same length and stack them.

        Args:
            arrays: List of arrays with shape (T_i, ...) where T_i can vary

        Returns:
            Stacked array with shape (num_arrays, max_T, ...)
        """
        max_len = max(arr.shape[0] for arr in arrays)
        padded_arrays = []

        for arr in arrays:
            if arr.shape[0] < max_len:
                # Pad by repeating the last frame
                pad_len = max_len - arr.shape[0]
                last_frame = arr[-1:].repeat(pad_len, axis=0)
                padded_arr = np.concatenate([arr, last_frame], axis=0)
            else:
                padded_arr = arr
            padded_arrays.append(padded_arr)

        return np.stack(padded_arrays, axis=0)

    def _process_trajectory_data(self) -> None:
        """Process raw trajectory data for RL training.

        Converts numpy arrays to torch tensors and computes velocities using
        Gaussian smoothing, following ManipTrans approach.
        """
        # Convert to torch tensors on device
        # Shape: (num_traj, max_T, ...)
        wrist_pos = torch.from_numpy(self._wrist_positions).float().to(self._device)
        wrist_rot_aa = torch.from_numpy(self._wrist_rotations_aa).float().to(self._device)
        dof_pos = torch.from_numpy(self._dof_positions).float().to(self._device)

        # Time delta for velocity computation
        time_delta = 1 / 60.0  # 60 Hz after skip=2 from 120 Hz

        # Compute velocities using ManipTrans methods with Gaussian smoothing
        # Functions expect (T, K, ...) format, we have (num_traj, max_T, ...)
        # Transpose to (max_T, num_traj, ...)
        num_traj, max_T, n_dofs = dof_pos.shape

        wrist_pos_T = wrist_pos.transpose(0, 1)  # (max_T, num_traj, 3)
        wrist_rot_aa_T = wrist_rot_aa.transpose(0, 1)  # (max_T, num_traj, 3)
        dof_pos_T = dof_pos.transpose(0, 1)  # (max_T, num_traj, n_dofs)

        wrist_vel_T = compute_velocity(wrist_pos_T, time_delta, gaussian_filter=True)  # (max_T, num_traj, 3)
        wrist_ang_vel_T = compute_angular_velocity(wrist_rot_aa_T, time_delta, gaussian_filter=True)  # (max_T, num_traj, 3)
        dof_vel_T = compute_dof_velocity(dof_pos_T, time_delta, gaussian_filter=True)  # (max_T, num_traj, n_dofs)

        # Transpose back to (num_traj, max_T, ...)
        wrist_vel = wrist_vel_T.transpose(0, 1)
        wrist_ang_vel = wrist_ang_vel_T.transpose(0, 1)
        dof_vel = dof_vel_T.transpose(0, 1)

        # Convert wrist rotations to quaternions (for easier manipulation)
        # Reshape for batch processing
        wrist_rot_aa_flat = wrist_rot_aa.reshape(-1, 3)
        angle = torch.norm(wrist_rot_aa_flat, dim=-1)  # (num_traj * max_T,)
        axis = wrist_rot_aa_flat / (angle.unsqueeze(-1) + 1e-8)  # (num_traj * max_T, 3)
        wrist_quat = quat_from_angle_axis(angle, axis)  # (num_traj * max_T, 4) [w, x, y, z]
        wrist_quat = wrist_quat.reshape(num_traj, max_T, 4)  # (num_traj, max_T, 4)

        # Load finger link's positions and velocities from MANO reference
        all_mano_joint_poses = []
        all_mano_joint_velocities = []
        all_mano_finger_tip_dists = []

        for traj_idx, demo_data in enumerate(self._demo_data_raw):
            mano_joint_poses = []
            mano_joint_velocities = []
            for joint_name in self._args.joint_mapping.keys():
                mano_joint_poses.append(
                    demo_data["mano_reference"]["finger_joints"][joint_name]  # (T, 3)
                )
                mano_joint_velocities.append(
                    demo_data["mano_reference"]["finger_joints_velocity"][joint_name]  # (T, 3)
                )
            mano_joint_poses = np.stack(mano_joint_poses, axis=1)  # (T, n_joints, 3)
            mano_joint_velocities = np.stack(mano_joint_velocities, axis=1)  # (T, n_joints, 3)

            all_mano_joint_poses.append(mano_joint_poses)
            all_mano_joint_velocities.append(mano_joint_velocities)
            all_mano_finger_tip_dists.append(
                demo_data["mano_reference"]["fingertip_distances"]  # (T, 5)
            )

        # Pad and stack MANO data
        mano_joint_poses_padded = self._pad_and_stack(all_mano_joint_poses)  # (num_traj, max_T, n_joints, 3)
        mano_joint_velocities_padded = self._pad_and_stack(all_mano_joint_velocities)  # (num_traj, max_T, n_joints, 3)
        mano_finger_tip_dists_padded = self._pad_and_stack(all_mano_finger_tip_dists)  # (num_traj, max_T, 5)

        self._mano_joint_poses = torch.from_numpy(mano_joint_poses_padded).float().to(self._device)
        self._mano_joint_velocities = torch.from_numpy(mano_joint_velocities_padded).float().to(self._device)
        self._mano_finger_tip_dists = torch.from_numpy(mano_finger_tip_dists_padded).float().to(self._device)

        # get the corresponding link idx for hand
        self._finger_link_idxs = []
        for link_name in self._args.joint_mapping.values():
            self._finger_link_idxs.append(self._robot.get_link(link_name).idx_local)

        # process object's pose and rotation
        obj_pos = torch.from_numpy(self._obj_pose_matrices[:, :, :3, 3]).float().to(self._device)  # (num_traj, max_T, 3)
        obj_rot_mat = torch.from_numpy(self._obj_pose_matrices[:, :, :3, :3]).float().to(self._device)  # (num_traj, max_T, 3, 3)
        obj_rot_quat = rotmat_to_quat(obj_rot_mat)  # (num_traj, max_T, 4)
        obj_rot_aa = quat_to_angle_axis(obj_rot_quat)  # (num_traj, max_T, 3)

        obj_pos_T = obj_pos.transpose(0, 1)  # (max_T, num_traj, 3)
        obj_rot_aa_T = obj_rot_aa.transpose(0, 1)  # (max_T, num_traj, 3)

        obj_vel_T = compute_velocity(obj_pos_T, time_delta, gaussian_filter=True)  # (max_T, num_traj, 3)
        obj_ang_vel_T = compute_angular_velocity(obj_rot_aa_T, time_delta, gaussian_filter=True)  # (max_T, num_traj, 3)

        obj_vel = obj_vel_T.transpose(0, 1)  # (num_traj, max_T, 3)
        obj_ang_vel = obj_ang_vel_T.transpose(0, 1)  # (num_traj, max_T, 3)

        # Store ORIGINAL processed trajectory data (per trajectory)
        # This will be used as a template for creating per-environment copies
        self._traj_data_template = {
            "wrist_pos": wrist_pos,  # (num_traj, max_T, 3)
            "wrist_rot": wrist_rot_aa,  # (num_traj, max_T, 3) axis-angle
            "wrist_quat": wrist_quat,  # (num_traj, max_T, 4) quaternion [w, x, y, z]
            "wrist_vel": wrist_vel,  # (num_traj, max_T, 3)
            "wrist_ang_vel": wrist_ang_vel,  # (num_traj, max_T, 3)
            "dof_pos": dof_pos,  # (num_traj, max_T, n_dofs)
            "dof_vel": dof_vel,  # (num_traj, max_T, n_dofs)
            "obj_pos": obj_pos,  # (num_traj, max_T, 3)
            "obj_quat": obj_rot_quat,  # (num_traj, max_T, 4)
            "obj_vel": obj_vel,  # (num_traj, max_T, 3)
            "obj_ang_vel": obj_ang_vel,  # (num_traj, max_T, 3)
        }

        # ===== Encode object mesh using BPS (Basis Point Set) =====
        # Following ManipTrans approach: load object mesh and encode to 128-dim BPS feature
        obj_id = self._args.object_id
        obj_mesh_path = f"{self._args.trajectory_path}/{obj_id}_collision.obj"

        # Load object mesh
        mesh = trimesh.load(obj_mesh_path, force='mesh')
        obj_verts = torch.from_numpy(mesh.vertices).float().to(self._device)  # (N_verts, 3)

        # Initialize BPS layer (same parameters as ManipTrans)
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere",   # Spherical grid of basis points
            n_bps_points=128,         # 128 basis points
            radius=0.2,               # 20cm radius sphere
            randomize=False,          # Fixed basis point locations
            device=self._device
        )

        # Encode object vertices to BPS feature
        # bps_layer.encode returns a dict with "dists" key containing the 128-dim feature
        obj_bps_single = self.bps_layer.encode(
            obj_verts.unsqueeze(0),  # Add batch dimension: (1, N_verts, 3)
            feature_type=self.bps_feat_type
        )[self.bps_feat_type].squeeze(0)  # (128,)

        # Repeat for all environments since all envs use the same object
        # This will be initialized properly in _init() after num_envs is known
        self._obj_bps_template = obj_bps_single  # Store template for later

        print(f"Object BPS encoding:")
        print(f"  - Mesh vertices: {obj_verts.shape}")
        print(f"  - BPS feature template: {obj_bps_single.shape}")

        print(f"Processed trajectory data:")
        print(f"  - Wrist positions: {wrist_pos.shape}")
        print(f"  - Wrist rotations (axis-angle): {wrist_rot_aa.shape}")
        print(f"  - Wrist rotations (quaternion): {wrist_quat.shape}")
        print(f"  - Wrist velocities: {wrist_vel.shape}")
        print(f"  - DOF positions: {dof_pos.shape}")
        print(f"  - DOF velocities: {dof_vel.shape}")
        print(f"  - MANO joint poses: {self._mano_joint_poses.shape}")
        print(f"  - MANO joint velocities: {self._mano_joint_velocities.shape}")

    def _augment_trajectory(self, envs_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply trajectory augmentation to specified environments.

        Following the augmentation strategy from replay_shadow_trajectory.py:
        1. Augment raw data (wrist position, wrist rotation axis-angle, MANO joint poses)
        2. Reuse processing logic to compute velocities and quaternions

        Args:
            envs_idx: Environment indices to augment (num_envs_to_aug,)

        Returns:
            Dictionary containing augmented trajectory data for these environments:
                - wrist_pos: (num_envs_to_aug, max_T, 3)
                - wrist_rot_aa: (num_envs_to_aug, max_T, 3)
                - wrist_quat: (num_envs_to_aug, max_T, 4)
                - wrist_vel: (num_envs_to_aug, max_T, 3)
                - wrist_ang_vel: (num_envs_to_aug, max_T, 3)
                - mano_joint_poses: (num_envs_to_aug, max_T, 20, 3)
                - mano_joint_velocities: (num_envs_to_aug, max_T, 20, 3)
        """
        assert False, "Not implemented for object yet!"

        num_envs = len(envs_idx)
        device = self.env_traj_idx.device

        # Get trajectory indices for these environments
        traj_idx = self.env_traj_idx[envs_idx]

        # Load original data from numpy arrays (not processed tensors)
        wrist_pos_orig = torch.from_numpy(
            self._wrist_positions[traj_idx.cpu().numpy()]
        ).float().to(device)  # (num_envs, max_T, 3)

        wrist_rot_aa_orig = torch.from_numpy(
            self._wrist_rotations_aa[traj_idx.cpu().numpy()]
        ).float().to(device)  # (num_envs, max_T, 3)

        mano_joint_poses_orig = self._mano_joint_poses[traj_idx]  # (num_envs, max_T, 20, 3)

        # ===== Sample augmentation parameters =====
        # 1. Random uniform scaling (applied only to wrist workspace, NOT to MANO joints)
        scale = torch.rand(num_envs, device=device) * (
            self._args.aug_scale_range[1] - self._args.aug_scale_range[0]
        ) + self._args.aug_scale_range[0]  # (num_envs,)

        # 2. Random translation (XY only, not Z to keep hand above ground)
        translation = torch.zeros(num_envs, 3, device=device)
        translation[:, 0] = torch.rand(num_envs, device=device) * (
            self._args.aug_translation_range[1] - self._args.aug_translation_range[0]
        ) + self._args.aug_translation_range[0]
        translation[:, 1] = torch.rand(num_envs, device=device) * (
            self._args.aug_translation_range[1] - self._args.aug_translation_range[0]
        ) + self._args.aug_translation_range[0]
        # translation[:, 2] = 0.0  # No vertical translation

        # 3. Random rotation around Z-axis
        theta_z = torch.rand(num_envs, device=device) * (
            self._args.aug_rotation_z_range[1] - self._args.aug_rotation_z_range[0]
        ) + self._args.aug_rotation_z_range[0]  # (num_envs,)

        # Build rotation matrices around Z-axis
        cos_z = torch.cos(theta_z)
        sin_z = torch.sin(theta_z)
        zeros = torch.zeros_like(cos_z)
        ones = torch.ones_like(cos_z)

        R_z = torch.stack([
            torch.stack([cos_z, -sin_z, zeros], dim=-1),
            torch.stack([sin_z, cos_z, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-2)  # (num_envs, 3, 3)

        # ===== Augment wrist positions =====
        # Apply scaling and rotation to wrist trajectory
        max_T = wrist_pos_orig.shape[1]

        # Reshape for batch matrix multiplication
        scaled_wrist = scale.unsqueeze(-1).unsqueeze(-1) * wrist_pos_orig  # (num_envs, max_T, 3)
        rotated_wrist = torch.bmm(
            R_z.unsqueeze(1).expand(-1, max_T, -1, -1).reshape(-1, 3, 3),
            scaled_wrist.reshape(-1, 3).unsqueeze(-1)
        ).squeeze(-1).reshape(num_envs, max_T, 3)  # (num_envs, max_T, 3)

        aug_wrist_pos = rotated_wrist + translation.unsqueeze(1)  # (num_envs, max_T, 3)

        # ===== Augment wrist rotations (axis-angle) =====
        # Apply rotation to wrist orientations (fully vectorized)
        # Convert all axis-angles to rotation matrices at once: (num_envs, max_T, 3) -> (num_envs, max_T, 3, 3)
        wrist_rot_mat = axis_angle_to_rotmat_batch(wrist_rot_aa_orig)  # (num_envs, max_T, 3, 3)

        # Apply Z-rotation: R_new = R_z @ R_wrist for all timesteps
        # Expand R_z: (num_envs, 3, 3) -> (num_envs, 1, 3, 3) -> (num_envs, max_T, 3, 3)
        R_z_expanded = R_z.unsqueeze(1).expand(-1, max_T, -1, -1)  # (num_envs, max_T, 3, 3)

        # Batch matrix multiply: (num_envs, max_T, 3, 3) @ (num_envs, max_T, 3, 3)
        aug_wrist_rot_mat = torch.matmul(R_z_expanded, wrist_rot_mat)  # (num_envs, max_T, 3, 3)

        # Convert back to axis-angle: (num_envs, max_T, 3, 3) -> (num_envs, max_T, 3)
        aug_wrist_rot_aa = rotmat_to_axis_angle_batch(aug_wrist_rot_mat)  # (num_envs, max_T, 3)

        # ===== Augment MANO joint poses =====
        # Compute offsets from original wrist (NOT scaled, only rotated)
        hand_to_joint = mano_joint_poses_orig - wrist_pos_orig.unsqueeze(2)  # (num_envs, max_T, 20, 3)

        # Apply rotation to offsets
        rotated_hand_to_joint = torch.bmm(
            R_z.unsqueeze(1).unsqueeze(1).expand(-1, max_T, 20, -1, -1).reshape(-1, 3, 3),
            hand_to_joint.reshape(-1, 3).unsqueeze(-1)
        ).squeeze(-1).reshape(num_envs, max_T, 20, 3)  # (num_envs, max_T, 20, 3)

        # New joint positions = new wrist + rotated offset
        aug_mano_joint_poses = aug_wrist_pos.unsqueeze(2) + rotated_hand_to_joint  # (num_envs, max_T, 20, 3)

        # ===== Compute velocities using same logic as _process_trajectory_data =====
        time_delta = 1 / 60.0  # 60 Hz after skip=2 from 120 Hz

        # Transpose to (max_T, num_envs, ...) for velocity computation
        wrist_pos_T = aug_wrist_pos.transpose(0, 1)  # (max_T, num_envs, 3)
        wrist_rot_aa_T = aug_wrist_rot_aa.transpose(0, 1)  # (max_T, num_envs, 3)

        wrist_vel_T = compute_velocity(wrist_pos_T, time_delta, gaussian_filter=True)  # (max_T, num_envs, 3)
        wrist_ang_vel_T = compute_angular_velocity(wrist_rot_aa_T, time_delta, gaussian_filter=True)  # (max_T, num_envs, 3)

        # Transpose back to (num_envs, max_T, ...)
        aug_wrist_vel = wrist_vel_T.transpose(0, 1)  # (num_envs, max_T, 3)
        aug_wrist_ang_vel = wrist_ang_vel_T.transpose(0, 1)  # (num_envs, max_T, 3)

        # ===== Convert wrist rotations to quaternions =====
        wrist_rot_aa_flat = aug_wrist_rot_aa.reshape(-1, 3)
        angle = torch.norm(wrist_rot_aa_flat, dim=-1)  # (num_envs * max_T,)
        axis = wrist_rot_aa_flat / (angle.unsqueeze(-1) + 1e-8)  # (num_envs * max_T, 3)
        aug_wrist_quat = quat_from_angle_axis(angle, axis)  # (num_envs * max_T, 4) [w, x, y, z]
        aug_wrist_quat = aug_wrist_quat.reshape(num_envs, max_T, 4)  # (num_envs, max_T, 4)

        # ===== Compute MANO joint velocities =====
        # Reshape to (max_T, num_envs * 20, 3) to compute all joints at once
        mano_joint_poses_T = aug_mano_joint_poses.permute(1, 0, 2, 3)  # (max_T, num_envs, 20, 3)
        mano_joint_poses_reshaped = mano_joint_poses_T.reshape(max_T, -1, 3)  # (max_T, num_envs*20, 3)

        # Compute velocities for all joints at once
        aug_mano_joint_vels_flat = compute_velocity(mano_joint_poses_reshaped, time_delta, gaussian_filter=True)  # (max_T, num_envs*20, 3)

        # Reshape back to (max_T, num_envs, 20, 3) then transpose to (num_envs, max_T, 20, 3)
        aug_mano_joint_vels = aug_mano_joint_vels_flat.reshape(max_T, num_envs, 20, 3).permute(1, 0, 2, 3)

        return {
            "wrist_pos": aug_wrist_pos,  # (num_envs, max_T, 3)
            "wrist_rot": aug_wrist_rot_aa,  # (num_envs, max_T, 3)
            "wrist_quat": aug_wrist_quat,  # (num_envs, max_T, 4)
            "wrist_vel": aug_wrist_vel,  # (num_envs, max_T, 3)
            "wrist_ang_vel": aug_wrist_ang_vel,  # (num_envs, max_T, 3)
            "mano_joint_poses": aug_mano_joint_poses,  # (num_envs, max_T, 20, 3)
            "mano_joint_velocities": aug_mano_joint_vels,  # (num_envs, max_T, 20, 3)
        }


    def _init(self) -> None:
        """Initialize observation and action spaces and environment buffers."""

        # Action space: hand is free-floating, so actions include base control (6D) + finger joints
        # Following ManipTrans: 6D wrist control (3D translation + 3D rotation) + n_dofs finger joints
        self._action_space = self._robot.action_space

        # Assign trajectories to environments in round-robin fashion
        # E.g., with 4 trajectories and 10 environments: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        self.env_traj_idx = torch.tensor(
            [i % self._num_trajectories for i in range(self.num_envs)],
            dtype=torch.long,
            device=self._device
        )

        # Per-environment trajectory lengths
        self.env_traj_lengths = torch.tensor(
            [self._traj_lengths[i % self._num_trajectories] for i in range(self.num_envs)],
            dtype=torch.long,
            device=self._device
        )

        # Create per-environment trajectory data (duplicated from template)
        # This allows each environment to have its own augmented trajectory
        max_T = self._max_traj_length
        self._traj_data = {
            "wrist_pos": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "wrist_rot": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "wrist_quat": torch.zeros((self.num_envs, max_T, 4), device=self._device),
            "wrist_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "wrist_ang_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "dof_pos": torch.zeros((self.num_envs, max_T, self._hand_dof_dim), device=self._device),
            "dof_vel": torch.zeros((self.num_envs, max_T, self._hand_dof_dim), device=self._device),
            "obj_pos": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "obj_quat": torch.zeros((self.num_envs, max_T, 4), device=self._device),
            "obj_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
            "obj_ang_vel": torch.zeros((self.num_envs, max_T, 3), device=self._device),
        }

        # Initialize with template data based on env_traj_idx
        for env_idx in range(self.num_envs):
            traj_idx = self.env_traj_idx[env_idx]
            for key in self._traj_data.keys():
                self._traj_data[key][env_idx] = self._traj_data_template[key][traj_idx]

        # Per-environment MANO joint data
        self.env_mano_joint_poses = torch.zeros((self.num_envs, max_T, 20, 3), device=self._device)
        self.env_mano_joint_velocities = torch.zeros((self.num_envs, max_T, 20, 3), device=self._device)
        self.env_mano_finger_tip_dists = torch.zeros((self.num_envs, max_T, 5), device=self._device)

        # Initialize with original data
        for env_idx in range(self.num_envs):
            traj_idx = self.env_traj_idx[env_idx]
            self.env_mano_joint_poses[env_idx] = self._mano_joint_poses[traj_idx]
            self.env_mano_joint_velocities[env_idx] = self._mano_joint_velocities[traj_idx]
            self.env_mano_finger_tip_dists[env_idx] = self._mano_finger_tip_dists[traj_idx]

        # Trajectory progress tracking
        # progress_buf: current position in trajectory timeline (for indexing trajectory data)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # Hand state buffers
        self.hand_dof_pos = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        self.hand_dof_vel = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)

        # Base state buffers (wrist pose)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self._device)  # [w, x, y, z]
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)

        # Additional buffers that matches ManipTrans structure
        self.cos_q = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        self.sin_q = torch.zeros((self.num_envs, self._hand_dof_dim), device=self._device)
        # Combine into base_state (10D: quat, lin_vel, ang_vel)
        self.base_state = torch.zeros((self.num_envs, 10), device=self._device)
        # Buffer related to target trajectory following
        K = self._obs_future_length
        self.target_wrist_pos = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.delta_wrist_pos = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.target_wrist_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.delta_wrist_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.target_wrist_quat = torch.zeros((self.num_envs, K * 4), device=self._device)
        self.delta_wrist_quat = torch.zeros((self.num_envs, K * 4), device=self._device)
        self.target_wrist_ang_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.delta_wrist_ang_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        # Buffer related to object target state
        self.target_object_pos = torch.zeros((self.num_envs, K, 3), device=self._device)
        self.delta_object_pos = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.target_object_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.delta_object_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.target_object_quat = torch.zeros((self.num_envs, K * 4), device=self._device)
        self.delta_object_quat = torch.zeros((self.num_envs, K * 4), device=self._device)
        self.target_object_ang_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.delta_object_ang_vel = torch.zeros((self.num_envs, K * 3), device=self._device)
        self.object_to_finger_tips = torch.zeros((self.num_envs, 5), device=self._device)  # 5 finger tips

        # Finger link position buffers for reward computation
        self.target_mano_joint_pos = torch.zeros((self.num_envs, K, 20, 3), device=self._device)
        self.finger_link_pos = torch.zeros((self.num_envs, 20, 3), device=self._device)
        self.target_mano_joint_vel = torch.zeros((self.num_envs, K * 20 * 3), device=self._device)
        self.finger_link_vel = torch.zeros((self.num_envs, 20, 3), device=self._device)
        self.delta_finger_link_pos = torch.zeros((self.num_envs, K * 20 * 3), device=self._device)
        self.delta_finger_link_vel = torch.zeros((self.num_envs, K * 20 * 3), device=self._device)
        self.mano_fingertip_to_object = torch.zeros((self.num_envs, K * 5), device=self._device)  # 5 finger tips

        # BPS object encoding buffer (repeated for all environments)
        # Shape: (num_envs, 128) - same BPS feature repeated across all envs
        self.bps = self._obj_bps_template.unsqueeze(0).repeat(self.num_envs, 1)  # (num_envs, 128)

        # Object state buffers
        self.object_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.object_pos_rel = torch.zeros((self.num_envs, 3), device=self._device)  # Relative to wrist base
        self.object_com_pos = torch.zeros((self.num_envs, 3), device=self._device)  # Center of mass position relative to wrist base
        self.object_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self.object_lin_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.object_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)

        # Environment state buffers
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)  # Episode time for warmup checks
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.success_count_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)

        # Action scale
        self._action_scale = self._args.robot_args.action_scale

        # Action history length and latency
        self._action_history_len = self._args.obs_history_len
        self._action_latency = self._args.action_latency

        # Action and DOF position buffers with history
        self._action_buf = torch.zeros(
            (self.num_envs, self._hand_dof_dim + self._base_dof_dim, self._action_history_len + 1), device=self._device
        )

        # Torques buffer for tracking
        self.dof_force = torch.zeros((self.num_envs, self._hand_dof_dim + self._base_dof_dim), device=self._device)

        # Rendering buffers
        self._rendered_images = []
        self._rendering = False
        self.camera_lookat = torch.tensor([0.0, 0.0, 0.4], device=self._device)
        self.camera_pos = torch.tensor([0.5, -0.5, 0.5], device=self._device)

        # Base policy observation buffer
        self._base_policy_obs = torch.zeros((self.num_envs, self._compute_base_policy_obs_dim()), device=self._device)

        # Tactile buffers (will be initialized after scene is built if tactile is enabled)
        # Only store z-axis (normal) force component
        if self._use_tactile:
            self._total_tactile_points = sum(
                config['num_points'] for config in self._tactile_sensor_configs.values()
            )
            self.tactile_forces_flat = torch.zeros((self.num_envs, self._total_tactile_points), device=self._device)
            # Note: We'll add tactile_forces_flat to observation terms dynamically in get_observations
            self.fingertip_max_force = torch.zeros((self.num_envs, 5), device=self._device)  # 5 finger tips
        else:
            self.tactile_forces_flat = torch.zeros((self.num_envs, 0), device=self._device)
            self.fingertip_max_force = torch.ones((self.num_envs, 5), device=self._device) * 1e-5

        # ===== Build observation spaces =====
        # Following ManipTrans architecture: three observation components

        # 1. Proprioception: q, cos(q), sin(q), base_state (ignore position), tactile
        prop_dim = (
            self._hand_dof_dim  # q
            + self._hand_dof_dim * 2  # cos(q) + sin(q)
            + 10  # base_state without position (quat + lin_vel + ang_vel)
        )
        if self._use_tactile:
            prop_dim += self._total_tactile_points  # tactile_forces_flat

        # 2. Privileged: dof_vel, object_state
        priv_dim = (
            self._hand_dof_dim  # hand_dof_vel
            + 3  # object_pos_rel
            + 4  # object_quat
            + 3  # object_lin_vel
            + 3  # object_ang_vel
            + 3  # object_com_pos
        )

        # 3. Target: reference trajectory information for next K timesteps
        # Following ManipTrans: delta_wrist_pos, wrist_vel, delta_wrist_vel, wrist_quat, delta_wrist_quat,
        #                        wrist_ang_vel, delta_wrist_ang_vel, delta_dof_pos, dof_vel, delta_dof_vel
        K = self._obs_future_length
        target_dim = (
            3 * K  # delta_wrist_pos
            + 3 * K  # wrist_vel
            + 3 * K  # delta_wrist_vel
            + 4 * K  # wrist_quat
            + 4 * K  # delta_wrist_quat
            + 3 * K  # wrist_ang_vel
            + 3 * K  # delta_wrist_ang_vel
            + self._hand_dof_dim * 3 * K  # delta finger_link_pos
            + self._hand_dof_dim * 3 * K  # finger link vel
            + self._hand_dof_dim * 3 * K  # delta finger_link_vel

            + 3 * K  # delta_object_pos
            + 3 * K  # target_object_vel
            + 3 * K  # delta_object_vel
            + 4 * K  # target_object_quat
            + 4 * K  # delta_object_quat
            + 3 * K  # target_object_ang_vel
            + 3 * K  # delta_object_ang_vel
            + 5  # object_to_finger_tips
            + 5 * K  # mano_fingertip_to_object
            + 128  # BPS object encoding
        )

        # Build observation spaces
        self._actor_observation_space = gym.spaces.Dict({
            "proprioception": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(prop_dim,), dtype=np.float32
            ),
            "privileged": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32
            ),
            "target": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32
            ),
        })

        self._critic_observation_space = gym.spaces.Dict({
            "proprioception": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(prop_dim,), dtype=np.float32
            ),
            "privileged": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(priv_dim,), dtype=np.float32
            ),
            "target": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(target_dim,), dtype=np.float32
            ),
        })

        self._info_space = gym.spaces.Dict({})
        self._extra_info = {}

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        """Reset specified environments.

        Following ManipTrans implementation:
        - Sample initial frame from trajectory (random or start)
        - Initialize DOF positions with noise from default pose
        - Initialize wrist pose from demo with noise
        - Reset all buffers
        """
        if len(envs_idx) == 0:
            return

        num_reset = len(envs_idx)

        # Get trajectory indices and lengths for the environments being reset
        reset_traj_idx = self.env_traj_idx[envs_idx]
        reset_traj_lengths = self.env_traj_lengths[envs_idx]

        if self._eval_mode:
            # if in eval mode, select reference trajectory in a rotation manner
            assert num_reset == 1, "Eval mode only supports resetting one env at a time."
            reset_traj_idx = (reset_traj_idx + 1) % self._num_trajectories
            reset_traj_lengths = torch.tensor(
                [self._traj_lengths[reset_traj_idx]],
                dtype=torch.long,
                device=self._device
            )
            self.env_traj_idx[envs_idx] = reset_traj_idx
            self.env_traj_lengths[envs_idx] = reset_traj_lengths

            for key in self._traj_data.keys():
                self._traj_data[key][envs_idx] = self._traj_data_template[key][reset_traj_idx]
            # Also update MANO data
            self.env_mano_joint_poses[envs_idx] = self._mano_joint_poses[reset_traj_idx]
            self.env_mano_joint_velocities[envs_idx] = self._mano_joint_velocities[reset_traj_idx]
            self.env_mano_finger_tip_dists[envs_idx] = self._mano_finger_tip_dists[reset_traj_idx]

        # ===== Apply trajectory augmentation if enabled =====
        if self._args.use_augmentation and not self._eval_mode:
            assert False, "Not implemented for object yet!"
            # TODO: implement trajectory augmentation for object case
            # Only augment trajectories if it has succeeded a certain number of times
            do_augment = self.success_count_buf[envs_idx] >= 100
            envs_to_augment = envs_idx[do_augment.cpu()]

            # Update success count buffer for environments that will be augmented
            self.success_count_buf[envs_to_augment] = 0

            if len(envs_to_augment) > 0:
                # Augment trajectories for the environments that meet the success threshold
                aug_traj_data = self._augment_trajectory(envs_to_augment)

                # Update trajectory data for these environments (using tensor indexing - no loop)
                self._traj_data["wrist_pos"][envs_to_augment] = aug_traj_data["wrist_pos"]
                self._traj_data["wrist_rot"][envs_to_augment] = aug_traj_data["wrist_rot"]
                self._traj_data["wrist_quat"][envs_to_augment] = aug_traj_data["wrist_quat"]
                self._traj_data["wrist_vel"][envs_to_augment] = aug_traj_data["wrist_vel"]
                self._traj_data["wrist_ang_vel"][envs_to_augment] = aug_traj_data["wrist_ang_vel"]
                self.env_mano_joint_poses[envs_to_augment] = aug_traj_data["mano_joint_poses"]
                self.env_mano_joint_velocities[envs_to_augment] = aug_traj_data["mano_joint_velocities"]

        # Sample initial timestep from trajectory
        # Following ManipTrans: random init or start from beginning
        random_state_init = getattr(self._args, 'random_state_init', False)
        if random_state_init and not self._eval_mode:
            # Random timestep initialization (up to 99% of trajectory length)
            # Use per-environment trajectory lengths
            seq_idx = torch.floor(
                reset_traj_lengths.float() * 0.99 * torch.rand(num_reset, device=self._device)
            ).long()
        else:
            # Always start from beginning
            seq_idx = torch.zeros(num_reset, dtype=torch.long, device=self._device)

        # Get DOF positions from trajectory at sampled timestep
        dof_pos = self._traj_data["dof_pos"][envs_idx, seq_idx]  # (num_reset, hand_dof_dim)
        dof_vel = self._traj_data["dof_vel"][envs_idx, seq_idx]  # (num_reset, hand_dof_dim)

        wrist_pos = self._traj_data["wrist_pos"][envs_idx, seq_idx]  # (num_reset, 3)
        wrist_quat = self._traj_data["wrist_quat"][envs_idx, seq_idx]  # (num_reset, 4)

        wrist_vel = self._traj_data["wrist_vel"][envs_idx, seq_idx]  # (num_reset, 3)
        wrist_ang_vel = self._traj_data["wrist_ang_vel"][envs_idx, seq_idx]  # (num_reset, 3)

        self._robot.reset_to_pose(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            wrist_pos=wrist_pos,
            wrist_quat=wrist_quat,
            base_dof_vel=torch.cat([wrist_vel, wrist_ang_vel], dim=-1),
            envs_idx=envs_idx,
        )

        object_pos = self._traj_data["obj_pos"][envs_idx, seq_idx]  # (num_reset, 3)
        object_quat = self._traj_data["obj_quat"][envs_idx, seq_idx]  # (num_reset, 4)
        object_lin_vel = self._traj_data["obj_vel"][envs_idx, seq_idx]  # (num_reset, 3)
        object_ang_vel = self._traj_data["obj_ang_vel"][envs_idx, seq_idx]  # (num_reset, 3)

        self._object.set_pos(object_pos, envs_idx=envs_idx)
        self._object.set_quat(object_quat, envs_idx=envs_idx)
        self._object.set_dofs_velocity(
            torch.cat([object_lin_vel, object_ang_vel], dim=-1),
            envs_idx=envs_idx
        )

        # ===== Reset all buffers =====
        self.progress_buf[envs_idx] = seq_idx  # Trajectory position
        self.time_since_reset[envs_idx] = 0.0  # Episode time
        self._action_buf[envs_idx] = 0.0
        self.reset_buf[envs_idx] = 0


    def get_terminated(self) -> torch.Tensor:
        """Check if episodes should terminate.

        Following ManipTrans termination logic:
        1. Error conditions (sanity checks for unrealistic velocities)
        2. Failed execution (tracking errors exceed thresholds after warmup)
        3. Success (reached near end of trajectory without failure)

        Note: No timeout truncation - episodes end naturally when trajectory completes or fails
        """
        # Initialize reset buffer
        reset_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # === Error conditions (sanity checks) ===
        # Following ManipTrans thresholds
        error_buf = (
            (torch.norm(self.base_lin_vel, dim=-1) > 100)
            | (torch.norm(self.base_ang_vel, dim=-1) > 200)
            | (torch.abs(self.hand_dof_vel).mean(-1) > 200)
            | (torch.norm(self.object_lin_vel, dim=-1) > 100)
            | (torch.norm(self.object_ang_vel, dim=-1) > 200)
        )

        # === Failed execution ===
        warmup_steps = 20
        warmup_time = warmup_steps * self._scene.scene.dt * self._args.robot_args.decimation
        warmup_done = self.time_since_reset >= warmup_time

        diff_joints_pos = self.target_mano_joint_pos[:, 0] - self.finger_link_pos  # (num_envs, 20, 3)
        diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)  # (num_envs, 20)

        diff_thumb_tip = diff_joints_pos_dist[:, 3]
        diff_index_tip = diff_joints_pos_dist[:, 7]
        diff_middle_tip = diff_joints_pos_dist[:, 11]
        diff_ring_tip = diff_joints_pos_dist[:, 15]
        diff_pinky_tip = diff_joints_pos_dist[:, 19]
        diff_level_1 = diff_joints_pos_dist[:, [0, 4, 8, 12, 16]].mean(dim=-1)
        diff_level_2 = diff_joints_pos_dist[:, [1, 2, 5, 6, 9, 10, 13, 14, 17, 18]].mean(dim=-1)

        diff_obj_pos_dist = torch.norm(
            self.target_object_pos[:, 0] - self.object_pos,
            dim=-1
        )  # (num_envs,)
        diff_obj_quat = quat_mul(
            self.target_object_quat[:, :4],
            quat_conjugate(self.object_quat)
        )  # (num_envs, 4)
        diff_obj_angle = 2 * torch.acos(
            torch.clamp(diff_obj_quat[:, 0], -1.0, 1.0)
        )
        diff_obj_angle = torch.abs(diff_obj_angle)

        mano_fingertip_to_object = self.mano_fingertip_to_object[:, :5]  # (num_envs, 5)

        # Scale factor for error thresholds (from ManipTrans, typically 0.7)
        scale_factor = 1.0
        failed_execute = (
            (diff_thumb_tip > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip > 0.05 / 0.7 * scale_factor)
            | (diff_ring_tip > 0.06 / 0.7 * scale_factor)
            | (diff_pinky_tip > 0.06 / 0.7 * scale_factor)
            | (diff_level_1 > 0.07 / 0.7 * scale_factor)
            | (diff_level_2 > 0.08 / 0.7 * scale_factor)
            | (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor**3)
            | (diff_obj_angle / np.pi * 180 > 30 / 0.343 * scale_factor**3)
            # | torch.any((mano_fingertip_to_object < 0.005) & (self.fingertip_max_force <= 0.0), dim=-1)
        ) & warmup_done

        # TODO: add finger tip distance failure condition

        # TODO: add curriculum learning with the scale factor

        failed_execute = failed_execute | error_buf

        # === Success condition ===
        # Reached within lookahead steps of trajectory end without failure
        lookahead = 3
        succeeded = (self.progress_buf + 1 + lookahead >= self.env_traj_lengths) & ~failed_execute

        # Update reset buffer with failure and success conditions
        reset_buf |= succeeded | failed_execute
        self.reset_buf[:] = reset_buf
        # Update success count buffer
        self.success_count_buf += succeeded.long()

        termination_dict = {
            "error_termination": error_buf,
            "failed_execution": failed_execute,
            "succeeded": succeeded,
            "any": reset_buf,
            "trajectory_updated": self.success_count_buf >= 100
        }
        self._extra_info["termination"] = termination_dict

        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        """Check if episodes should truncate due to time limit.

        For trajectory following tasks, we don't use timeout truncation.
        Episodes end naturally when trajectory is completed or fails.
        """
        time_out_buf = torch.zeros_like(self.time_out_buf, dtype=torch.bool)
        self.time_out_buf[:] = time_out_buf

        # We do progress update here
        # Because there will be a wrapper calling this env, and it will not call step() to update progress
        self.progress_buf += 1

        return time_out_buf

    def update_buffers(self) -> None:
        """Update all state buffers from simulator."""
        # Update hand state
        self.hand_dof_pos[:] = self._robot.get_dofs_position(dofs_idx_local=self._robot._hand_dof_idx_local)
        self.hand_dof_vel[:] = self._robot.get_dofs_velocity(dofs_idx_local=self._robot._hand_dof_idx_local)

        # Update base state (wrist pose)
        self.base_pos[:] = self._robot.base_pos
        self.base_quat[:] = self._robot.base_quat
        self.base_lin_vel[:] = self._robot.base_lin_vel
        self.base_ang_vel[:] = self._robot.base_ang_vel
        # Update object state
        self.object_pos[:] = self._object.get_pos()
        self.object_pos_rel[:] = self.object_pos - self.base_pos
        self.object_com_pos[:] = self._object.get_links_pos(
            links_idx_local=0, ref="link_com"
        ).squeeze(1) - self.base_pos
        self.object_quat[:] = self._object.get_quat()
        self.object_lin_vel[:] = self._object.get_vel()
        self.object_ang_vel[:] = self._object.get_ang()

        # Update ManipTrans-style buffers
        self.cos_q[:] = torch.cos(self.hand_dof_pos)
        self.sin_q[:] = torch.sin(self.hand_dof_pos)
        # Combine into base_state: [quat, lin_vel, ang_vel]
        self.base_state[:] = torch.cat([
            self.base_quat,
            self.base_lin_vel,
            self.base_ang_vel,
        ], dim=-1)

        # Update target trajectory following buffers
        K = self._obs_future_length
        cur_idx = self.progress_buf  # (num_envs,) Current timestep in trajectory
        future_indices = cur_idx.unsqueeze(-1) + torch.arange(1, K + 1, device=self._device).unsqueeze(0)
        # Clamp using per-environment trajectory lengths
        # Use torch.min to clamp the max per environment
        max_indices = (self.env_traj_lengths.unsqueeze(-1) - 1).expand(-1, K)  # (num_envs, K)
        future_indices = torch.clamp(future_indices, min=0)
        future_indices = torch.min(future_indices, max_indices)  # (num_envs, K)

        # Index trajectory data using environment index and timestep
        env_indices = torch.arange(self.num_envs, device=self._device).unsqueeze(-1).expand(-1, K)  # (num_envs, K)

        # Update wrist target states
        target_wrist_pos = self._traj_data["wrist_pos"][env_indices, future_indices]  # (num_envs, K, 3)
        target_wrist_quat = self._traj_data["wrist_quat"][env_indices, future_indices]  # (num_envs, K, 4)
        target_wrist_vel = self._traj_data["wrist_vel"][env_indices, future_indices]  # (num_envs, K, 3)
        target_wrist_ang_vel = self._traj_data["wrist_ang_vel"][env_indices, future_indices]  # (num_envs, K, 3)

        self.target_wrist_pos[:] = target_wrist_pos.reshape(self.num_envs, -1)
        self.delta_wrist_pos[:] = (target_wrist_pos - self.base_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        self.target_wrist_vel[:] = target_wrist_vel.reshape(self.num_envs, -1)
        self.delta_wrist_vel[:] = (target_wrist_vel - self.base_lin_vel.unsqueeze(1)).reshape(self.num_envs, -1)
        self.target_wrist_quat[:] = target_wrist_quat.reshape(self.num_envs, -1)
        self.delta_wrist_quat[:] = quat_mul(
            target_wrist_quat, 
            quat_conjugate(self.base_quat.unsqueeze(1).expand(-1, K, -1))
        ).reshape(self.num_envs, -1)
        self.target_wrist_ang_vel[:] = target_wrist_ang_vel.reshape(self.num_envs, -1)
        self.delta_wrist_ang_vel[:] = (target_wrist_ang_vel - self.base_ang_vel.unsqueeze(1)).reshape(self.num_envs, -1)

        # Update object target states
        self.target_object_pos[:] = self._traj_data["obj_pos"][env_indices, future_indices]  # (num_envs, K, 3)
        target_object_quat = self._traj_data["obj_quat"][env_indices, future_indices]  # (num_envs, K, 4)
        target_object_vel = self._traj_data["obj_vel"][env_indices, future_indices]  # (num_envs, K, 3)
        target_object_ang_vel = self._traj_data["obj_ang_vel"][env_indices, future_indices]  # (num_envs, K, 3)

        self.delta_object_pos[:] = (self.target_object_pos - self.object_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        self.target_object_vel[:] = target_object_vel.reshape(self.num_envs, -1)
        self.delta_object_vel[:] = (target_object_vel - self.object_lin_vel.unsqueeze(1)).reshape(self.num_envs, -1)
        self.target_object_quat[:] = target_object_quat.reshape(self.num_envs, -1)
        self.delta_object_quat[:] = quat_mul(
            target_object_quat,
            quat_conjugate(self.object_quat.unsqueeze(1).expand(-1, K, -1))
        ).reshape(self.num_envs, -1)
        self.target_object_ang_vel[:] = target_object_ang_vel.reshape(self.num_envs, -1)
        self.delta_object_ang_vel[:] = (target_object_ang_vel - self.object_ang_vel.unsqueeze(1)).reshape(self.num_envs, -1)

        # Compute object to finger tip distances
        fingertip_pos = self._robot.fingertip_pos  # (num_envs, 5, 3)
        self.object_to_finger_tips[:] = torch.norm(
            self.object_pos.unsqueeze(1) - fingertip_pos,
            dim=-1
        )  # (num_envs, 5)

        # Update MANO joint poses
        self.target_mano_joint_pos[:] = self.env_mano_joint_poses[env_indices, future_indices]  # (num_envs, K, 20, 3)
        self.finger_link_pos[:] = self._robot.get_links_pos(links_idx_local=self._finger_link_idxs)  # (num_envs, 20, 3)
        self.finger_link_vel[:] = self._robot.get_links_vel(links_idx_local=self._finger_link_idxs)  # (num_envs, 20, 3)
        target_mano_joint_vel = self.env_mano_joint_velocities[env_indices, future_indices]  # (num_envs, K, 20, 3)
        self.delta_finger_link_pos[:] = (self.target_mano_joint_pos - self.finger_link_pos.unsqueeze(1)).reshape(self.num_envs, -1)
        self.delta_finger_link_vel[:] = (target_mano_joint_vel - self.finger_link_vel.unsqueeze(1)).reshape(self.num_envs, -1)
        self.target_mano_joint_vel[:] = target_mano_joint_vel.reshape(self.num_envs, -1)

        self.mano_fingertip_to_object[:] = self.env_mano_finger_tip_dists[env_indices, future_indices].reshape(self.num_envs, -1)  # (num_envs, K * 5)

        # Update tactile sensor data if enabled (only z-axis / normal force)
        if self._use_tactile:
            tactile_forces_list = []
            fingertip_max_force = []
            for link_name in sorted(self._tactile_sensors.keys()):
                sensor = self._tactile_sensors[link_name]
                config = self._tactile_sensor_configs[link_name]

                # Read sensor data (returns num_points * 3 forces)
                force_field_full = sensor.read()
                num_points = config['num_points']

                # Reshape to (num_points, 3) to extract z-axis component
                force_field_3d = force_field_full.reshape(self.num_envs, num_points, 3)

                # Extract only z-axis (normal) force: shape (num_envs, num_points)
                force_z = force_field_3d[..., 2]

                tactile_forces_list.append(force_z)
                fingertip_max_force.append(torch.max(force_z, dim=-1).values.unsqueeze(-1))  # (num_envs, 1)

            # Concatenate all sensor readings: shape (num_envs, total_tactile_points)
            self.tactile_forces_flat[:] = torch.cat(tactile_forces_list, dim=-1)
            self.fingertip_max_force[:] = torch.cat(fingertip_max_force, dim=-1)  # (num_envs, 5)

            # Update visualization if enabled
            if self._tactile_visualizer is not None:
                self._tactile_visualizer.update()

        self.dof_force[:] = self._robot.dofs_control_force


    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get actor and critic observations."""
        self.update_buffers()

        # Build actor observation from configured terms
        obs_components = []
        for key in self._args.actor_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_noise = torch.randn_like(obs_gt) * self._args.obs_noises.get(key, 0.0)
            if self._eval_mode:
                obs_noise *= 0
            obs_components.append(obs_gt + obs_noise)
        actor_obs = torch.cat(obs_components, dim=-1)

        # Build critic observation from configured terms
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)

        # Build base policy observation if residual learning is enabled
        if self._use_residual_learning:
            obs_components = []
            for key in self._base_policy_obs_terms:
                obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
                obs_noise = torch.randn_like(obs_gt) * self._args.obs_noises.get(key, 0.0)
                if self._eval_mode:
                    obs_noise *= 0
                obs_components.append(obs_gt + obs_noise)
            self._base_policy_obs[:] = torch.cat(obs_components, dim=-1)

        return actor_obs, critic_obs

    def _pre_step(self) -> None:
        """Update timers before each physics substep."""
        dt = self._scene.scene.dt
        self.time_since_reset += dt

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to the environment using delta position control.

        Following inhand_rotation_env.py pattern:
        - Actions are deltas (not absolute positions)
        - For free base: first 6 DOFs are base (3 pos + 3 rot), rest are fingers
        - Apply different scaling to base and finger actions
        - Target position = current position + scaled delta

        For residual learning:
        - If enabled, compute base policy action from precomputed base policy observations
        - Add residual action (from training policy) to base action
        - Use combined action for control
        """
        action = action.detach().to(self._device)

        # ===== Residual learning: compute base action and add residual =====
        if self._use_residual_learning:
            # Get base action from base policy (deterministic, using precomputed observations)
            with torch.no_grad():
                base_action, _ = self._base_policy(self._base_policy_obs, deterministic=True)

            # Add residual action to base action
            action = base_action + action

        # Update action history buffer (shift and add new action)
        self._action_buf[:] = torch.cat([self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1)

        # Get action from history buffer with latency
        exec_action = self._action_buf[:, :, -self._action_latency - 1]

        # For free base hand: first 6 DOFs are base (3 pos + 3 rot), rest are fingers
        # Apply different scaling to base and finger actions
        base_action_scale = 0.01  # Smaller scale for base position/rotation
        finger_action_scale = self._action_scale

        # Scale base actions (first 6 DOFs: translation + rotation)
        exec_action[:, :6] *= base_action_scale
        # Scale finger actions (remaining DOFs)
        exec_action[:, 6:] *= finger_action_scale

        # Target position = current + scaled delta
        # For free base, hand_dof_pos includes both base (6D) and fingers (20D)
        target_dof_pos = self._robot.dof_pos + exec_action

        # TEMPT
        # env_indices = torch.arange(self.num_envs, device=self._device)
        # target_hand_dof = self._traj_data["dof_pos"][env_indices, self.progress_buf + 1]
        # target_base_dof = self._robot.dof_pos[:, :6]
        # target_dof_pos = torch.cat([target_base_dof, target_hand_dof], dim=-1)


        # Apply actions and simulate physics with decimation
        for _ in range(self._args.robot_args.decimation):
            self._pre_step()

            self._robot.apply_action(action=target_dof_pos)
            self._scene.scene.step()
        
        # TEMPT
        # target_base_pos = self._traj_data["wrist_pos"][env_indices, self.progress_buf + 1]
        # target_base_quat = self._traj_data["wrist_quat"][env_indices, self.progress_buf + 1]
        # self._robot.set_pos(target_base_pos)
        # self._robot.set_quat(target_base_quat)
        # target_obj_pos = self._traj_data["obj_pos"][env_indices, self.progress_buf + 1]
        # target_obj_quat = self._traj_data["obj_quat"][env_indices, self.progress_buf + 1]
        # self._object.set_pos(target_obj_pos)
        # self._object.set_quat(target_obj_quat)

        self.update_buffers()

        # Render if rendering is enabled
        self._render_headless()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment."""
        # Apply action
        self.apply_action(action)

        # Get terminated
        terminated = self.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)

        # Get truncated
        truncated = self.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)

        # Get reward
        reward, reward_terms = self.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)

        # Get extra infos
        extra_infos = self.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms

        # Reset if terminated or truncated
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)

        # Get observations
        next_obs, _ = self.get_observations()

        return next_obs, reward, terminated, truncated, extra_infos

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute trajectory following rewards using modular reward terms."""
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}

        # Prepare state dict for reward functions
        state_dict = {key: getattr(self, key) for key in self._reward_required_keys}

        # Compute all configured rewards and sum them
        # Each reward term already has its scale applied in the RewardTerm class
        for key, func in self._reward_functions.items():
            reward = func(state_dict)
            reward_total += reward
            reward_dict[f"{key}"] = reward.clone()

        reward_dict["Total"] = reward_total

        return reward_total, reward_dict
    
    def update_history(self) -> None:
        """Update action history buffer after each step."""
        # This function can be expanded if additional history updates are needed
        pass

    def get_extra_infos(self) -> dict[str, Any]:
        self.update_buffers()
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        obs_tensor = torch.cat(obs_components, dim=-1)
        self._extra_info["observations"] = {"critic": obs_tensor}
        self._extra_info["time_outs"] = self.time_out_buf.clone()[:, None]
        return self._extra_info

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_dim(self) -> int:
        return self._hand_dof_dim + self._base_dof_dim  # finger DOFs + 6D wrist control

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._actor_observation_space)

    @property
    def critic_obs_dim(self) -> int:
        return get_space_dim(self._critic_observation_space)

    @property
    def scene(self) -> FlatScene:
        return self._scene

    @property
    def robot(self) -> WUJIHand:
        return self._robot

    @property
    def dt(self) -> float:
        """Environment timestep (accounts for decimation)."""
        return self._scene.scene.dt * self._args.robot_args.decimation

    def _render_headless(self) -> None:
        """Render a frame from the floating camera if rendering is enabled."""
        if self._rendering and len(self._rendered_images) < 1000:
            hand_pos = self.base_pos[0]
            self._floating_camera.set_pose(
                pos=hand_pos + self.camera_pos,
                lookat=hand_pos + self.camera_lookat,
            )
            rgb, _, _, _ = self._floating_camera.render()
            self._rendered_images.append(rgb)

    def start_rendering(self) -> None:
        """Start recording rendered images."""
        self._rendering = True
        self._rendered_images = []

    def stop_rendering(self, save_gif: bool = True, gif_path: str = ".") -> None:
        """Stop recording and optionally save as GIF."""
        self._rendering = False
        if save_gif and self._rendered_images:
            self.save_gif(gif_path)

    def save_gif(self, gif_path: str, duration: int = 100) -> None:
        """Save the rendered images as a GIF."""
        if not self._rendered_images:
            print("No rendered images to save.")
            return

        pil_images = []
        for img_array in self._rendered_images:
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)

        if pil_images:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=0,
            )
            print(f"GIF saved to: {gif_path}")

    def eval(self) -> None:
        """Set environment to evaluation mode."""
        self._eval_mode = True

    def close(self) -> None:
        """Clean up resources."""
        if self._tactile_visualizer is not None:
            self._tactile_visualizer.close()
            self._tactile_visualizer = None

    def _setup_tactile_sensors(self, args: SingleHandRetargetingEnvArgs) -> None:
        """Setup tactile sensors from JSON grid file."""
        import json

        if args.tactile_grid_path is None:
            raise ValueError("tactile_grid_path must be specified when use_tactile=True")

        tactile_grid_path = Path(args.tactile_grid_path)
        if not tactile_grid_path.exists():
            raise FileNotFoundError(f"Tactile grid file not found: {tactile_grid_path}")

        print(f"\n{'='*70}")
        print(f"Loading tactile grid from: {tactile_grid_path}")
        print(f"{'='*70}")

        with open(tactile_grid_path, 'r') as f:
            tactile_data = json.load(f)

        links_data = tactile_data['links']
        sensor_link_names = list(links_data.keys())

        print(f"Configuring tactile sensors for links: {sensor_link_names}")

        # Validate sensor links and store tactile points
        for link_name in sensor_link_names:
            if link_name not in links_data:
                raise ValueError(f"Link '{link_name}' not found in tactile grid JSON")

            link_data = links_data[link_name]
            points = link_data.get('points', [])

            if not points:
                raise ValueError(f"No tactile points found for link '{link_name}'")

            # Store local positions
            local_positions = np.array(points, dtype=np.float32)
            self._tactile_points[link_name] = local_positions

            print(f"  {link_name}: {len(local_positions)} tactile points")

        # Add TactileFieldSensors
        print(f"\n{'='*70}")
        print(f"Adding TactileFieldSensors")
        print(f"{'='*70}")

        for link_name in sensor_link_names:
            link_data = links_data[link_name]
            link_idx_local = self.robot.get_link(link_name).idx_local
            assert link_idx_local == link_data['link_idx_local'], f"Link index mismatch for {link_name}: {link_idx_local} vs {link_data['link_idx_local']}"
            num_points = link_data['num_points']
            local_positions = self._tactile_points[link_name]

            print(f"\nSensor for {link_name}:")
            print(f"  Link index (local): {link_idx_local}")
            print(f"  Tactile points: {num_points}")

            # Create TactileFieldSensor with custom tactile points
            sensor = self._scene.scene.add_sensor(
                gs.sensors.TactileField(
                    entity_idx=self._robot._robot_entity.idx,
                    link_idx_local=link_idx_local,
                    indenter_entity_idx=self._object.idx,
                    indenter_link_idx_local=0,
                    tactile_points_local=local_positions,
                    kn=args.tactile_kn,
                )
            )

            self._tactile_sensors[link_name] = sensor
            self._tactile_sensor_configs[link_name] = {
                'num_points': num_points,
                'link_idx_local': link_idx_local,
            }

            print(f"  ✓ TactileFieldSensor added")
