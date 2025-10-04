import os
import argparse

TASK_LIST = [
    # ('anymal', 'walk'),
    # ('anymal', 'walk1'),
    # ('anymal', 'walk2'),
    # ('g1', 'hop'),
    # ('g1', 'hop1'),
    # ('g1', 'hop2'),
    # ('g1', 'hop3'),
    # ('g1', 'hop4'),
    # ('go2', 'handstand'),
    # ('go2', 'handstand1'),
    # ('go2', 'handstand2'),
    # ('h1_2', 'walk'),
    # ('h1_2', 'walk1'),
    # ('h1_2', 'walk2'),
    # ('leap_hand', 'walk'),
    # ('leap_hand', 'walk1'),
    # ('leap_hand', 'walk2'),
    # ('shadow_hand', 'walk'),
    # ('shadow_hand', 'walk1'),
    # ('shadow_hand', 'walk2'),
    # ('lamp', 'hop'),
    # ('lamp', 'hop1'),
    # ('lamp', 'hop2'),
    # ('tripod', 'walk'),
    # ('tripod', 'walk1'),
    # ('tripod', 'walk2'),
]

# parser = argparse.ArgumentParser()
# parser.add_argument('-r', '--robot', type=str, default='leap_hand')
# parser.add_argument('-n', '--name', type=str, default='default')
# args = parser.parse_args()

# os.system(f'python agent_pose.py -r {args.robot} -n {args.name}')
# os.system(f'python agent_dof_pos.py -r {args.robot} -n {args.name}')
# os.system(f'python adjust_gait.py -r {args.robot} -n {args.name}')

for robot, name in TASK_LIST:
    # os.system(f'python agent_pose.py -r {robot} -n {name}')
    # os.system(f'python agent_dof_pos.py -r {robot} -n {name}')
    # os.system(f'python adjust_gait.py -r {robot} -n {name}')
    os.system(f'python build_final_cfg.py -r {robot} -n {name}')
