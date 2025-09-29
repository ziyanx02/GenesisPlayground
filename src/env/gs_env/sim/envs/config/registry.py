from gs_env.sim.envs.config.schema import EnvArgs, GenesisInitArgs, LeggedRobotEnvArgs
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
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_default"],
    robot_args=RobotArgsRegistry["franka_teleop"],
    objects_args=[ObjectArgsRegistry["box_default"]],
    sensors_args=[],
    reward_args={},
    img_resolution=(480, 270),
)


EnvArgsRegistry["walk_default"] = LeggedRobotEnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["flat_scene_legged"],
    robot_args=RobotArgsRegistry["g1_no_waist"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1_no_waist",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 100.0,
        "AngVelZReward": 100.0,
        "LinVelZPenalty": 2.0,
        "AngVelXYPenalty": 1.0,
        ### Pose Tracking ###
        "OrientationPenalty": 50.0,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "ActionRatePenalty": 0.1,
        "DofPosLimitPenalty": 100.0,
        "G1BaseHeightPenalty": 300.0,
        "ActionLimitPenalty": 1.0,
        ### Motion Constraints ###
        "AnkleTorquePenalty": 0.001,
        "HipYawPenalty": 10.0,
        "UpperBodyDofPenalty": 10.0,
        "FeetAirTimeReward": 1.0,
        "G1FeetHeightPenalty": 1.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
)


EnvArgsRegistry["custom_desk"] = LeggedRobotEnvArgs(
    gs_init_args=GenesisInitArgsRegistry["default"],
    scene_args=SceneArgsRegistry["custom_scene_desk"],
    robot_args=RobotArgsRegistry["g1_default"],
    objects_args=[],
    sensors_args=[],
    reward_term="g1",
    reward_args={
        ### Velocity Tracking ###
        "LinVelXYReward": 10.0,
        "AngVelZReward": 10.0,
        "LinVelZPenalty": 2.0,
        "AngVelXYPenalty": 1.0,
        ### Pose Tracking ###
        "OrientationPenalty": 100.0,
        ### Regularization ###
        "TorquePenalty": 0.0001,
        "ActionRatePenalty": 0.1,
        "DofPosLimitPenalty": 100.0,
        "G1BaseHeightPenalty": 300.0,
        "ActionLimitPenalty": 1.0,
        ### Motion Constraints ###
        "AnkleTorquePenalty": 0.0001,
        "HipYawPenalty": 1.0,
        "UpperBodyDofPenalty": 3.0,
        "FeetAirTimeReward": 10.0,
    },
    img_resolution=(480, 270),
    action_latency=1,
    obs_history_len=1,
)
