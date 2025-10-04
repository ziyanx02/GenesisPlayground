import os
import yaml
import argparse
import threading

import numpy as np
import cv2
import pickle

from robot_display.display import Display
from prompts.prompts import *
from api.azure_openai import complete, local_image_to_data_url
from agent import Agent, VisOptions

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
args = parser.parse_args()

cfg_path = f"./cfgs/{args.robot}/{args.name}.yaml"
cfg = yaml.safe_load(open(cfg_path))

log_dir = os.path.dirname(os.path.abspath(__file__)) + f"/logs/{args.robot}/{args.name}_pose"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

vis_options = VisOptions()
vis_options.merge_fixed_links = cfg["robot"]["merge_fixed_links"]
vis_options.show_viewer = False
agent = Agent(cfg_path, vis_options=vis_options)

visible_links_id, camera_transforms = agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
agent.render(log_dir=log_dir)

task = cfg["task"]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": ROOT_SELECTION_PROMPT.format(visible_links_id=visible_links_id, task=task)},
]

for axis in ["x", "y", "z"]:
    image_path = f"{log_dir}/label_{axis}.png"
    messages.append(
        {
            "role": "user",
            "content": [
                        {"type": "text", "text": ORIENTATION_PROMPT[axis]},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(image_path)},
                        },
                    ],
        }
    )

print('###### Get Body_link_id ######')
response = complete(messages)
print(response)
body_link_id = int(response)

def rotate_robot(agent, axis, angle):
    if axis == "x":
        agent.rotate_along_x(angle)
    if axis == "y":
        agent.rotate_along_y(angle)
    if axis == "z":
        agent.rotate_along_z(angle)

agent.set_body_link(body_link_id)
agent.update()
agent.render(log_dir=log_dir)
agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
agent.render_from_nx(agent.get_body_pos(), log_dir=log_dir)

max_attempts = 5  # avoid infinite loops

rotation_propose_messages = [{"role": "system", "content": SYSTEM_PROMPT},]

for attempt in range(max_attempts):
    # Save pre-rotation image
    print("###### Save pre-rotation image of xyz & cam view ######")
    agent.render(log_dir=log_dir)
    agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
    agent.render_from_nx(agent.get_body_pos(), log_dir=log_dir)
    before_images = {}
    for axis in ["-x", "y", "z"]:
        before_images[axis] = local_image_to_data_url(f"{log_dir}/label_{axis}.png")

    # Ask VLM to propose a rotation
    rotation_propose_messages.append({"role": "user", "content": ROTATION_PROPOSE_PROMPT.format(task=task, body_link_id=body_link_id)})
    for axis in ["-x", "y", "z"]:
        rotation_propose_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": ORIENTATION_PROMPT[axis]},
                {"type": "image_url", "image_url": {"url": before_images[axis]}}
            ]
        })

    print(f"###### Propose Rotation of attempt {attempt} ######")
    response = complete(rotation_propose_messages)
    print(response)
    rotation_propose_messages.append({"role": "assistant", "content": response})

    lines = response.split("Answer:")[-1].strip().split("\n")
    if lines[0].strip().lower() == "no":
        break

    axis = lines[1].strip().lower()
    angle = float(lines[2].strip())

    # Perform rotation
    rotate_robot(agent, axis, angle)
    agent.update()

    # Render new image and get feedback
    print("###### Save post-rotation image of xyz & cam view ######")
    agent.render(log_dir=log_dir)
    agent.render_from_xyz(agent.get_body_pos(), log_dir=log_dir)
    agent.render_from_nx(agent.get_body_pos(), log_dir=log_dir)
    after_images = {}
    for ax in ["-x", "y", "z"]:
        after_images[ax] = local_image_to_data_url(f"{log_dir}/label_{ax}.png")

    # Ask VLM to evaluate the rotation
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ROTATION_EVALUATE_PROMPT.format(task=task, axis=axis, angle=angle)},
    ]
    for ax in ["-x", "y", "z"]:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Before rotation. " + ORIENTATION_PROMPT[ax]},
                {"type": "image_url", "image_url": {"url": before_images[ax]}},
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "After rotation. " + ORIENTATION_PROMPT[ax]},
                {"type": "image_url", "image_url": {"url": after_images[ax]}}
            ]
        })

    print("###### Evaluate Rotation ######")
    response = complete(messages)
    print(response)
    answer = response.split("Answer:")[-1].strip().lower()
    rotation_propose_messages.append({"role": "user", "content": response})

    if "cancel" in answer:
        # Undo last rotation
        print(f"Cancelling last rotation: -{angle} along {axis}")
        rotate_robot(agent, axis, -angle)
        agent.update()
    elif "done" in answer:
        break
    else:
        continue

links = agent.display.links

cfg["base_link_name"] = links[body_link_id].name
cfg["body_init_quat"] = agent.get_body_quat().numpy().tolist()
cfg["base_init_quat"] = agent.get_link_quat(0).numpy().tolist()
yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_pose.yaml", "w"))
with open(f"{log_dir}/pose.pkl", "wb") as f:
    pickle.dump(rotation_propose_messages, f)

print("Finished")
