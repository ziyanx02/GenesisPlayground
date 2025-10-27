from gs_env.sim.envs.config.schema import (
    EnvArgs,
    GenesisInitArgs,
    LeggedRobotEnvArgs,
    ManipulationEnvArgs,
    MotionEnvArgs,
    WalkingEnvArgs,
)
from gs_env.sim.objects.config.registry import ObjectArgsRegistry
from gs_env.sim.robots.config.registry import RobotArgsRegistry
from gs_env.sim.scenes.config.registry import SceneArgsRegistry
from gs_env.sim.sensors.config.registry import SensorArgsRegistry

# ------------------------------------------------------------
# Genesis init
# ------------------------------------------------------------


GenesisInitArgsRegistry: dict[str, GenesisInitArgs] = {}


GenesisInitArgsRegistry["default"] = GenesisInitArgs(
    seed=0,
    precision="32",
    logging_level="info",
    backend=None,
)


# ------------------------------------------------------------
# Manipulation
# ------------------------------------------------------------


EnvArgsRegistry: dict[str, EnvArgs] = {}

EnvArgsRegistry["goal_reach_default"] = EnvArgs(
    env_name="GoalReachingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_default"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[
        SensorArgsRegistry["oak_camera_default"],
        SensorArgsRegistry["ee_link_pos"],
        SensorArgsRegistry["ee_link_quat"],
        SensorArgsRegistry["joint_angles"],
        SensorArgsRegistry["gripper_width"],
    ],
    reward_term="reward",
    reward_args={
        "ActionL2Penalty": 0.0,
        "KeypointsAlign": 1.0,
    },
    img_resolution=(480, 270),
)


EnvArgsRegistry["pick_cube_default"] = EnvArgs(
    env_name="GoalReachingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
)

# ------------------------------------------------------------
# G1 Configuration
# ------------------------------------------------------------

EnvArgsRegistry["g1_walk"] = WalkingEnvArgs(
    env_name="WalkingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 10.0,
        "AngVelZReward": 10.0,
        "LinVelZPenalty": 20.0,
        "AngVelXYPenalty": 1.0,
        ### Pose Tracking ###
        "OrientationPenalty": 100.0,
        ### Regularization ###
        "TorquePenalty": 0.00001,
        "ActionRatePenalty": 0.3,
        "DofPosLimitPenalty": 10.0,
        # "G1BaseHeightPenalty": 30.0,
        "ActionLimitPenalty": 0.1,
        ### Motion Constraints ###
        "AnkleTorquePenalty": 0.001,
        "HipYawPenalty": 10.0,
        "HipRollPenalty": 100.0,
        # "UpperBodyDofPenalty": 3.0,
        "UpperBodyActionPenalty": 0.5,
        "WaistDofPenalty": 300.0,
        # "FeetAirTimeReward": 200.0,
        "FeetAirTimePenalty": 500.0,
        "G1FeetHeightPenalty": 100.0,
        "G1FeetContactForcePenalty": 30.0,
        "FeetZVelocityPenalty": 30.0,
        "FeetOrientationPenalty": 30.0,
        "StandStillFeetContactPenalty": 3e-4,
        "StandStillActionRatePenalty": 1.0,
        # "StandStillAnkleTorquePenalty": 0.01,
        "G1FeetContactForceLimitPenalty": 1e-4,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "base_ang_vel": 0.5,
        "feet_contact_force": 0.001,
    },
    obs_noises={
        "dof_pos": 0.01,
        "dof_vel": 0.2,
        "projected_gravity": 0.05,
        "base_ang_vel": 0.2,
    },
    actor_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_ang_vel",
        "commands",
    ],
    critic_obs_terms=[
        "last_action",
        "dof_pos",
        "dof_vel",
        "projected_gravity",
        "base_lin_vel",
        "base_ang_vel",
        "commands",
        "feet_height",
        "feet_contact_force",
    ],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    command_resample_time=10.0,
    commands_range=(
        (-1.0, 1.0),  # Forward/Backward
        (0.0, 0.0),  # Left/Right
        (-1.0, 1.0),  # Turn
    ),
)


EnvArgsRegistry["g1_motion"] = MotionEnvArgs(
    env_name="MotionEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Motion Tracking ###
        "DofPosReward": 10.0,
        "DofVelReward": 2.0,
        # "BaseHeightReward": 1.0,
        "BasePosReward": 10.0,
        "BaseQuatReward": 10.0,
        "BaseLinVelReward": 20.0,
        "BaseAngVelReward": 20.0,
        # "KeyBodyPosReward": 10.0,
        ### Regularization ###
        "TorquePenalty": 0.00001,
        "ActionRatePenalty": 0.3,
        "DofPosLimitPenalty": 10.0,
        "ActionLimitPenalty": 0.1,
        "AnkleTorquePenalty": 0.001,
        "AngVelXYPenalty": 1.0,
        "G1FeetContactForceLimitPenalty": 1e-4,
        "FeetAirTimePenalty": 100.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={
        "dof_vel": 0.1,
        "base_ang_vel_local": 0.5,
        "feet_contact_force": 0.001,
    },
    obs_noises={
        "dof_pos": 0.01,
        "dof_vel": 0.2,
        "projected_gravity": 0.05,
        "base_ang_vel_local": 0.2,
    },
    actor_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Reference
        "ref_dof_pos",
        "ref_dof_vel",
        "ref_base_euler",
        "ref_base_lin_vel_local",
        "ref_base_ang_vel_local",
        "ref_base_rotation_6D",
    ],
    critic_obs_terms=[
        "last_action",
        # Proprioception
        "dof_pos",
        "dof_vel",
        "base_euler",
        "base_lin_vel_local",
        "base_ang_vel_local",
        "base_rotation_6D",
        "projected_gravity",
        # Reference
        "ref_dof_pos",
        "ref_dof_vel",
        "ref_base_euler",
        "ref_base_lin_vel_local",
        "ref_base_ang_vel_local",
        "ref_base_rotation_6D",
        # Privilleged
        "feet_contact_force",
    ],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
    motion_file="assets/motion/twist_dataset.yaml",
)


EnvArgsRegistry["g1_fixed"] = LeggedRobotEnvArgs(
    env_name="WalkingEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_fixed"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1_no_waist",
    reward_args={},
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={},
    obs_noises={},
    actor_obs_terms=[],
    critic_obs_terms=[],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
)


EnvArgsRegistry["custom_g1_mocap"] = LeggedRobotEnvArgs(
    env_name="CustomEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_g1_mocap"],
    robot_args=RobotArgsRegistry["g1_fixed"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1_no_waist",
    reward_args={},
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
    obs_scales={},
    obs_noises={},
    actor_obs_terms=[],
    critic_obs_terms=[],
    terminate_after_collision_on=[
        "pelvis",
        "torso_link",
        "left_hip_yaw_link",
        "right_hip_yaw_link",
        "left_knee_link",
        "right_knee_link",
        "left_shoulder_yaw_link",
        "right_shoulder_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
    ],
)

# ------------------------------------------------------------
# WUJI Hand In-Hand Rotation Configuration
# ------------------------------------------------------------

EnvArgsRegistry["wuji_inhand_rotation"] = ManipulationEnvArgs(
    env_name="InHandRotationEnv",
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["wuji_hand"],
    objects_args=[],  # Cube is created in environment
    sensors_args=[],
    reward_term="manipulation",
    reward_args={
        ### Main Task Rewards (Penspin-style) ###
        "RotateRewardClipped": {
            "scale": 1.0,  # rotate_reward_scale from penspin
            "angvel_clip_min": -0.5,  # angvelClipMin from penspin
            "angvel_clip_max": 0.5,  # angvelClipMax from penspin
        },
        "RotatePenaltyThreshold": {
            "scale": 0.03,  # rotate_penalty_scale from penspin (0.3)
            "angvel_penalty_threshold": 1.0,  # angvelPenaltyThres from penspin
        },
        ### Regularization Penalties ###
        "ObjectLinVelPenalty": {
            "scale": 0.003,  # obj_linvel_penalty_scale from penspin (0.3)
        },
        "PoseDiffPenalty": {
            "scale": 0.01,  # pose_diff_penalty_scale from penspin (0.1)
        },
        "TorquePenalty": {
            "scale": 0.01,  # torque_penalty_scale from penspin (0.1)
        },
        "WorkPenalty": {
            "scale": 0.001,  # work_penalty_scale from penspin (1.0)
        },
        "PositionPenalty": {
            "scale": 0.01,  # position_penalty_scale from penspin (0.1)
            "target_x": 0.0,  # target position from penspin (line 551-552)
            "target_y": 0.0,
            "target_z": 0.23,  # Adjusted for WUJI hand height (penspin uses reset_z_threshold + 0.01)
        },
        # Note: penspin also uses pencil_z_dist_penalty_scale: -1.0, but we use cube not pencil
        # Note: penspin also tracks finger_obj_penalty but doesn't include it in reward
    },
    cube_args={
        "size": 0.07,
        "position": (0.0, 0.0, 0.22),
    },
    img_resolution=(480, 480),
    action_latency=1,
    obs_history_len=3,  # 3 timesteps of history
    obs_scales={
        "hand_dof_vel": 0.1,
        "cube_ang_vel": 0.5,
    },
    obs_noises={
        "hand_dof_pos": 0.01,
        "hand_dof_vel": 0.1,
        "cube_pos": 0.002,
        "cube_quat": 0.01,
    },
    actor_obs_terms=[
        # Action history (20 * 3 = 60D)
        "action_history_flat",  # Flattened action history
        # DOF position history (20 * 3 = 60D)
        "dof_pos_history_flat",  # Flattened DOF position history
    ],
    critic_obs_terms=[
        # Action history (20 * 3 = 60D)
        "action_history_flat",
        # DOF position history (20 * 3 = 60D)
        "dof_pos_history_flat",
        # Hand velocity (20D)
        "hand_dof_vel",
        # Cube state (3 + 4 + 3 + 3 = 13D)
        "cube_pos",
        "cube_quat",
        "cube_lin_vel",
        "cube_ang_vel",
    ],
)
