import argparse
import yaml
import time

from pose_estimation.robot_display.display import Display
from utils.low_state_handler import LowStateMsgHandler
from gs_env.sim.envs.config.registry import EnvArgsRegistry

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='g1')
parser.add_argument('-c', '--cfg', type=str, default='walk_default')
args = parser.parse_args()

cfg = EnvArgsRegistry[args.cfg]

robot = Display(cfg)

low_state_handler = LowStateMsgHandler(cfg)
low_state_handler.init()

while True:
    time.sleep(0.1)
    robot.set_body_quat(low_state_handler.quat)
    robot.set_dofs_position(low_state_handler.joint_pos)
    robot.update()