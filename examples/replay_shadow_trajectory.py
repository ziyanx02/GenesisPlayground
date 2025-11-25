import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs


def augment_trajectory(
    shadow_traj,
    object_traj,
    mano_reference,
    translation_range=(-0.1, 0.1),
    rotation_z_range=(-np.pi / 4, np.pi / 4),
    uniform_scale_range=(0.9, 1.1),
    seed=None,
):
    """
    Augment hand-object trajectory while preserving physical plausibility.

    Applies three types of augmentation:
    1. Random translation (XYZ) - applied to all positions
    2. Random rotation around Z-axis - applied to all orientations and positions
    3. Uniform scaling - applied only to wrist and object positions (NOT to MANO joints or finger DOFs)

    Args:
        shadow_traj: Dictionary containing hand trajectory data
        object_traj: Dictionary containing object trajectory data
        mano_reference: Dictionary containing MANO reference joint positions
        translation_range: (min, max) for random translation in meters for each axis
        rotation_z_range: (min, max) for random rotation around Z-axis in radians
        uniform_scale_range: (min, max) for uniform scaling of wrist/object positions only
        seed: Random seed for reproducibility

    Returns:
        Tuple of (augmented_shadow_traj, augmented_object_traj, augmented_mano_reference)
    """
    if seed is not None:
        np.random.seed(seed)

    # Deep copy to avoid modifying original data
    aug_shadow_traj = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in shadow_traj.items()}
    aug_object_traj = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in object_traj.items()}
    aug_mano_reference = {
        k: {kk: vv.copy() if isinstance(vv, np.ndarray) else vv for kk, vv in v.items()}
        if isinstance(v, dict) else (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in mano_reference.items()
    }

    # 1. Random uniform scaling (applied only to wrist and object, NOT to MANO joints)
    scale = np.random.uniform(uniform_scale_range[0], uniform_scale_range[1])

    # 2. Random translation (apply after scaling) - only horizontal (X, Y), not vertical (Z)
    translation = np.array([
        np.random.uniform(translation_range[0], translation_range[1]),  # X
        np.random.uniform(translation_range[0], translation_range[1]),  # Y
        0.0  # Z - no vertical translation to keep hand above ground
    ])

    # 3. Random rotation around Z-axis
    theta_z = np.random.uniform(rotation_z_range[0], rotation_z_range[1])
    cos_z, sin_z = np.cos(theta_z), np.sin(theta_z)
    R_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ])

    # Get original data
    wrist_positions = aug_shadow_traj["wrist_positions"]  # (T, 3)
    object_positions = aug_object_traj["positions"]  # (T, 3)
    object_pose_matrices = aug_object_traj["pose_matrices"]  # (T, 4, 4)

    # Compute relative offsets - these will be preserved (NOT scaled)
    hand_to_object = object_positions - wrist_positions  # (T, 3)

    # Apply scaling and rotation to wrist trajectory (workspace size)
    scaled_wrist = scale * wrist_positions
    rotated_wrist = (R_z @ scaled_wrist.T).T
    aug_shadow_traj["wrist_positions"] = rotated_wrist + translation

    # Apply rotation to wrist orientations (axis-angle)
    wrist_rotations_aa = aug_shadow_traj["wrist_rotations_aa"]
    for t in range(len(wrist_rotations_aa)):
        # Convert axis-angle to rotation matrix
        aa = wrist_rotations_aa[t]
        angle = np.linalg.norm(aa)
        if angle > 1e-6:
            R_wrist = R.from_rotvec(aa).as_matrix()
        else:
            R_wrist = np.eye(3)

        # Apply Z-rotation: R_new = R_z @ R_wrist
        R_new = R_z @ R_wrist
        # Convert back to axis-angle
        aug_shadow_traj["wrist_rotations_aa"][t] = R.from_matrix(R_new).as_rotvec()

    # DOF positions (finger joint angles) remain unchanged - they are intrinsic
    # aug_shadow_traj["dof_positions"] stays the same

    # Compute new object positions: new_wrist + rotated(original hand_to_object offset)
    rotated_hand_to_object = (R_z @ hand_to_object.T).T
    aug_object_traj["positions"] = aug_shadow_traj["wrist_positions"] + rotated_hand_to_object

    # Update object pose matrices
    for t in range(len(object_pose_matrices)):
        pose = object_pose_matrices[t].copy()

        # Use the new object position
        new_obj_pos = aug_object_traj["positions"][t]

        # Apply rotation to object orientation
        obj_rot = pose[:3, :3]
        new_obj_rot = R_z @ obj_rot

        # Update pose matrix
        aug_object_traj["pose_matrices"][t, :3, 3] = new_obj_pos
        aug_object_traj["pose_matrices"][t, :3, :3] = new_obj_rot

    # Update MANO reference finger joints: new_wrist + rotated(original hand_to_joint offset)
    if "finger_joints" in aug_mano_reference:
        for joint_name in aug_mano_reference["finger_joints"]:
            joint_positions = aug_mano_reference["finger_joints"][joint_name]  # (T, 3)

            # Compute original relative offset from wrist to joint (NOT scaled)
            hand_to_joint = joint_positions - wrist_positions

            # Rotate the offset (but do NOT scale it)
            rotated_hand_to_joint = (R_z @ hand_to_joint.T).T

            # New joint position = new wrist position + rotated (unscaled) offset
            aug_mano_reference["finger_joints"][joint_name] = (
                aug_shadow_traj["wrist_positions"] + rotated_hand_to_joint
            )

    return aug_shadow_traj, aug_object_traj, aug_mano_reference


def main():
    parser = argparse.ArgumentParser(description="Replay Shadow Hand trajectory")
    parser.add_argument(
        "--trajectory",
        type=str,
        default="output_trajectories_mujoco/wujihand_hand_trajectory_07bb1@0_mujoco.pkl",
        help="Path to trajectory pickle file",
    )
    parser.add_argument(
        "--object-mesh",
        type=str,
        default="output_trajectories_mujoco/scan_coacd.obj",
        help="Path to object mesh file",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = normal speed)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the trajectory playback",
    )
    parser.add_argument(
        "--save-render",
        type=str,
        default=None,
        help="Path to save rendered video",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply random augmentation to the trajectory",
    )
    parser.add_argument(
        "--aug-translation",
        type=float,
        default=0.1,
        help="Range for random translation (±meters) in each axis",
    )
    parser.add_argument(
        "--aug-rotation-z",
        type=float,
        default=45.0,
        help="Range for random Z-axis rotation (±degrees)",
    )
    parser.add_argument(
        "--aug-scale-min",
        type=float,
        default=0.9,
        help="Minimum uniform scale factor",
    )
    parser.add_argument(
        "--aug-scale-max",
        type=float,
        default=1.1,
        help="Maximum uniform scale factor",
    )
    parser.add_argument(
        "--aug-seed",
        type=int,
        default=None,
        help="Random seed for augmentation (for reproducibility)",
    )
    args = parser.parse_args()

    ########################## Load trajectory data ##########################
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    print(f"\n{'='*80}")
    print(f"Loading trajectory from: {trajectory_path}")
    print(f"{'='*80}")

    with open(trajectory_path, "rb") as f:
        data = pickle.load(f)

    # Extract trajectory data
    shadow_traj = data["hand_trajectory"]
    object_traj = data["object_trajectory"]
    mano_reference = data["mano_reference"]
    metadata = data["metadata"]

    wrist_positions = shadow_traj["wrist_positions"]  # (T, 3)
    wrist_rotations = shadow_traj["wrist_rotations_aa"]  # (T, 3) axis-angle
    dof_positions = shadow_traj["dof_positions"]  # (T, 22)

    object_positions = object_traj["positions"]  # (T, 3)
    object_pose_matrices = object_traj["pose_matrices"]  # (T, 4, 4)

    # wrist_positions[:, 2] += 0.4
    # object_pose_matrices[:, 2, 3] += 0.4

    T = len(wrist_positions)
    print(f"Trajectory length: {T} timesteps")
    print(f"Shadow DOF count: {dof_positions.shape[1]}")
    print(f"{'='*80}\n")

    ########################## Apply augmentation if requested ##########################
    if args.augment:
        print(f"{'='*80}")
        print("Applying trajectory augmentation...")
        print(f"  Translation range: ±{args.aug_translation:.3f}m")
        print(f"  Z-rotation range: ±{args.aug_rotation_z:.1f}°")
        print(f"  Scale range: [{args.aug_scale_min:.2f}, {args.aug_scale_max:.2f}]")
        if args.aug_seed is not None:
            print(f"  Random seed: {args.aug_seed}")
        print(f"{'='*80}")

        shadow_traj, object_traj, mano_reference = augment_trajectory(
            shadow_traj,
            object_traj,
            mano_reference,
            translation_range=(-args.aug_translation, args.aug_translation),
            rotation_z_range=(-np.deg2rad(args.aug_rotation_z), np.deg2rad(args.aug_rotation_z)),
            uniform_scale_range=(args.aug_scale_min, args.aug_scale_max),
            seed=args.aug_seed,
        )

        # Re-extract augmented data
        wrist_positions = shadow_traj["wrist_positions"]
        wrist_rotations = shadow_traj["wrist_rotations_aa"]
        dof_positions = shadow_traj["dof_positions"]
        object_positions = object_traj["positions"]
        object_pose_matrices = object_traj["pose_matrices"]

        print("Augmentation applied successfully!\n")

    ########################## init ##########################
    gs.init(backend=gs.cpu)

    ########################## create a scene ##########################
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.5, -0.5, 0.5),
        camera_lookat=(0.0, 0.0, 0.4),
        camera_fov=40,
        max_FPS=60,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0, 0, 0),
            enable_collision=False,
        ),
        show_viewer=True,
    )

    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # Add object mesh
    load_data_dir = "/".join(str(trajectory_path).split("/")[:-1])
    obj_id = metadata['obj_id']
    object_mesh_path = Path(f"{load_data_dir}/") / Path(f"{obj_id}_collision.obj")
    print(f"Loading object mesh from: {object_mesh_path}")
    # Apply 90-degree rotation around X-axis to correct mesh orientation
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(object_mesh_path),
            pos=object_positions[0],
            euler=(90, 0, 0),  # Rotate 90 degrees around X-axis
        ),
    )
    
    joint_markers = {}
    for joint_name in mano_reference["finger_joints"].keys():
        joint_markers[joint_name] = scene.add_entity(
            gs.morphs.Sphere(
                radius=0.005,
                pos=mano_reference["finger_joints"][joint_name][0],
            )
    )
    link_markers = {}
    link_names = [
        # "finger1_link1", "finger2_link1", "finger3_link1", "finger4_link1", "finger5_link1",
        "finger1_link2", "finger2_link2", "finger3_link2", "finger4_link2", "finger5_link2",
        "finger1_link3", "finger2_link3", "finger3_link3", "finger4_link3", "finger5_link3",
        "finger1_link4", "finger2_link4", "finger3_link4", "finger4_link4", "finger5_link4",
        "finger1_tip_link", "finger2_tip_link", "finger3_tip_link", "finger4_tip_link", "finger5_tip_link"
    ]
    # for link_name in link_names:
    #     link_markers[link_name] = scene.add_entity(
    #         gs.morphs.Box(
    #             size=(0.01, 0.01, 0.01),
    #             pos=[0, 0, 0],
    #         )
    #     )

    # Add Shadow Hand
    shadow_hand = scene.add_entity(
        gs.morphs.URDF(
            file="assets/robot/wujihand-urdf/urdf/right.urdf",
            merge_fixed_links=False,
            fixed=False,
            is_free=True,
        ),
        vis_mode="collision"
    )

    ########################## camera setup ##########################
    cam = None
    if args.save_render:
        cam = scene.add_camera(
            res=(1920, 1080),
            pos=(0.5, -0.5, 0.5),
            lookat=(0.0, 0.0, -0.3),
            fov=40,
            GUI=False,
        )

    ########################## build ##########################
    scene.build()

    joint_names = ["finger1_joint1", "finger1_joint2", "finger1_joint3", "finger1_joint4",
                   "finger2_joint1", "finger2_joint2", "finger2_joint3", "finger2_joint4",
                   "finger3_joint1", "finger3_joint2", "finger3_joint3", "finger3_joint4",
                   "finger4_joint1", "finger4_joint2", "finger4_joint3", "finger4_joint4",
                   "finger5_joint1", "finger5_joint2", "finger5_joint3", "finger5_joint4"]
    # Get DOF indices for finger joints (excluding forearm_joint and wrist_joint)
    finger_dofs_idx = []
    for name in joint_names:
        joint = shadow_hand.get_joint(name)
        # Each joint has 1 DOF
        finger_dofs_idx.extend(joint.dofs_idx_local)
    links_idx_local = {name: shadow_hand.get_link(name).idx_local for name in link_names}

    print(f"Shadow Hand total DOFs: {shadow_hand.n_dofs}")
    print(f"Finger DOFs mapped: {len(finger_dofs_idx)}")
    print(f"Trajectory finger DOFs: {dof_positions.shape[1]}")

    if len(finger_dofs_idx) != dof_positions.shape[1]:
        print(f"WARNING: DOF count mismatch! Mapped {len(finger_dofs_idx)} but trajectory has {dof_positions.shape[1]}")

    # Start camera recording if camera was added
    if cam is not None:
        cam.start_recording()

    ########################## Replay trajectory ##########################
    def replay():
        while True:
            for t in range(0, T, max(1, int(1 / args.speed))):
                # Convert axis-angle rotation to quaternion
                axis_angle = wrist_rotations[t]
                angle = np.linalg.norm(axis_angle)
                if angle > 1e-6:
                    axis = axis_angle / angle
                    # Both angle and axis need to be numpy arrays
                    angle_array = np.array(angle)
                    quat = gs.utils.geom.axis_angle_to_quat(angle_array, axis)
                else:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])

                # Set Shadow Hand base pose (wrist position and rotation)
                wrist_pos = wrist_positions[t]
                # Set base position and orientation using set_pos and set_quat
                shadow_hand.set_pos(wrist_pos)
                shadow_hand.set_quat(quat)

                # Set finger joint positions
                finger_dofs = dof_positions[t]
                shadow_hand.set_dofs_position(finger_dofs, finger_dofs_idx)

                # Set object pose from 4x4 matrix
                pose_matrix = object_pose_matrices[t]
                obj_pos = pose_matrix[:3, 3]
                obj_rot_matrix = pose_matrix[:3, :3]
                obj_quat = gs.utils.geom.R_to_quat(obj_rot_matrix)

                obj.set_qpos(np.concatenate([obj_pos, obj_quat]))

                # Update joint marker positions
                for joint_name, marker in joint_markers.items():
                    marker_pose = mano_reference["finger_joints"][joint_name][t]
                    marker.set_pos(marker_pose)
                # Update link marker positions
                # for name, marker in link_markers.items():
                #     link_pos = shadow_hand.get_links_pos(links_idx_local=links_idx_local[name])
                #     marker.set_pos(link_pos[0])

                # Step scene (no physics, just visualization)
                scene.step()

                # Render camera frame if recording
                if cam is not None:
                    cam.render()

                # Progress indicator
                if t % 50 == 0:
                    progress = (t / T) * 100
                    print(f"Progress: {progress:.1f}% ({t}/{T})", end="\r")

            print(f"\nTrajectory replay completed ({T} steps)")

            if not args.loop:
                break

        # Stop camera recording if it was started
        if cam is not None:
            cam.stop_recording(save_to_filename=args.save_render, fps=30)
            print(f"\nVideo saved to: {args.save_render}")

    replay()


if __name__ == "__main__":
    main()
