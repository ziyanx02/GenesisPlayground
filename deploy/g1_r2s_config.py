G1_HEIGHT = [0, 0, 0.793]
G1_CB1_POS = [-2.0, 0.0, 0.793]
G1_CB1_QUAT = [1.0, 0.0, 0.0, 0.0]

G1_TRACKED_LINK_NAMES = [
    "pelvis_contour_link",
    "torso_link",
    "left_rubber_hand",
    "right_rubber_hand",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

G1_CB1_LINK_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

G1_CB2_LINK_NAMES = [
    "pelvis_contour_link",
    "torso_link",
    "left_rubber_hand",
    "right_rubber_hand",
    # "right_ankle_roll_link",
]

G1_IK_TABLES = {
    "pelvis_contour_link": [
        "pelvis",
    ],
    "torso_link": [
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
    "left_rubber_hand": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        # "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "right_rubber_hand": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        # "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "left_ankle_roll_link": [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ],
    "right_ankle_roll_link": [
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ],
}
