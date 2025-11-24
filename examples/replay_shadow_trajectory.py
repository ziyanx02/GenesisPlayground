import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import genesis as gs


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
    for link_name in link_names:
        link_markers[link_name] = scene.add_entity(
            gs.morphs.Box(
                size=(0.01, 0.01, 0.01),
                pos=[0, 0, 0],
            )
        )

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
                # for link_name, marker in link_markers.items():
                #     link_pose = shadow_hand.get_link(link_name).get_pos()
                #     marker.set_pos(link_pose)
                for name, marker in link_markers.items():
                    link_pos = shadow_hand.get_links_pos(links_idx_local=links_idx_local[name])
                    marker.set_pos(link_pos[0])

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
