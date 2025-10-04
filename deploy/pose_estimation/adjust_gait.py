import os
import yaml
import argparse
import threading

import numpy as np
import torch
import cv2
from PIL import Image
from scipy.ndimage import label, center_of_mass
import matplotlib.pyplot as plt
import pickle

from robot_display.display import Display
from api.azure_openai import complete, local_image_to_data_url
from prompts.prompts import *
from agent import Agent, VisOptions

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/{args.name}_dof_pos.yaml"
cfg = yaml.safe_load(open(cfg_path))

log_dir = os.path.dirname(os.path.abspath(__file__)) + f"/logs/{args.robot}/{args.name}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

vis_options = VisOptions()
vis_options.merge_fixed_links = cfg["robot"]["merge_fixed_links"]
vis_options.show_viewer = False
agent = Agent(cfg_path, vis_options=vis_options)

visible_links_id, camera_transforms = agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
agent.render(log_dir=log_dir)

task = cfg["task"]

body_link_id = agent.display.get_link_by_name(cfg["base_link_name"]).idx_local
agent.set_body_link(body_link_id)
agent.set_body_quat(cfg["body_init_quat"])
extremities = cfg["extremity ids"]
dof_pos = agent.display.dof_pos
joint_name_to_dof_order = agent.display.joint_name_to_dof_order
for joint_name in cfg["dof_names"]:
    dof_pos[joint_name_to_dof_order[joint_name]] = cfg["default_joint_angles"][joint_name]
agent.display.set_dofs_position(dof_pos)
agent.update()

agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
agent.render_from_nxyz(agent.get_body_pos(), log_dir=log_dir)
agent.render(log_dir=log_dir)

prompt = GAIT_DESIGN_PROMPT.format(visible_links_id=visible_links_id, extremities=extremities, task=task)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt},
]

for orientation in ["x", "-y", "z"]:
    image_path = f"{log_dir}/label_{orientation}.png"
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": ORIENTATION_PROMPT[orientation]},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )
response = complete(messages)
print(response)

lines = response.split("\n")
for line_id in range(len(lines)):
    if "Answer:" in lines[line_id]:
        lines = lines[line_id + 1:]
        break
for line_id in range(len(lines)):
        if "feet:" in lines[line_id]:
            feet = [int(num.strip()) for num in lines[line_id + 1].split(",")]
        if "duration:" in lines[line_id]:
            duration = [float(num.strip()) for num in lines[line_id + 1].split(",")]
        if "phase:" in lines[line_id]:
            phase = [float(num.strip()) for num in lines[line_id + 1].split(",")]
        if "frequency:" in lines[line_id]:
            frequency = [float(num.strip()) for num in lines[line_id + 1].split(",")]

feet_pos = [agent.get_link_pos(link_id) for link_id in feet]
body_pos = agent.get_body_pos()
base_height_target = (body_pos[2] - torch.mean(torch.cat([foot_pos.unsqueeze(0) for foot_pos in feet_pos])[:, 2])).item()

feet_link_names = [agent.display.links[foot_id].name for foot_id in feet]
stationary_position = [foot_pos[:2].numpy().tolist() for foot_pos in feet_pos]
feet_height_target = [0.2 * base_height_target for _ in feet_link_names]

gait_cfg = {}
gait_cfg["base_height_target"] = base_height_target
gait_cfg["frequency"] = frequency
gait_cfg["duration"] = duration
gait_cfg["offset"] = phase
gait_cfg["stationary_position"] = stationary_position
cfg["diameter"] = float(agent.display.diameter)
cfg["mass"] = float(agent.display.mass)
cfg["link_names"] = [link.name for link in agent.display.links]
cfg["gait"] = gait_cfg
cfg["feet_link_names"] = feet_link_names

base_init_pos = agent.get_link_pos(0)
base_init_pos[:2] = 0
base_init_pos[2] -= agent.display.entity.get_AABB()[0, 2].item()
base_init_pos[2] *= 1.2
body_init_pos = base_init_pos + agent.get_body_pos() - agent.get_link_pos(0)
cfg["base_init_pos"] = base_init_pos.tolist()
cfg["body_init_pos"] = body_init_pos.tolist()

yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_gait.yaml", "w"))