import os
import yaml
import argparse
import threading

from robot_display.gui_display import GUIDisplay

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='go2')
parser.add_argument('-n', '--name', type=str, default=None)
args = parser.parse_args()

cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))
if args.name is not None:
    cfg = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}.yaml"))
display = GUIDisplay(
    cfg=cfg,
    body_pos=False,
    body_pose=False,
    dofs_pos=True,
    pd_control=True,
)
display.run()