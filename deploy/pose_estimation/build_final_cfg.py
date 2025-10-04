import os
import copy
import yaml
import argparse

'''
Combine generated cfg with pre-defined control & training parameters
'''

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--robot', type=str, default='leap_hand')
parser.add_argument('-n', '--name', type=str, default='default')
args = parser.parse_args()

cfg_gait = yaml.safe_load(open(f"./cfgs/{args.robot}/{args.name}_gait.yaml"))
cfg_control = yaml.safe_load(open(f"./cfgs/{args.robot}/basic.yaml"))

cfg = cfg_control

cfg['robot']['body_name'] = cfg_gait['base_link_name']
cfg['robot']['foot_names'] = cfg_gait['feet_link_names']
cfg['robot']['links_to_keep'] = copy.copy(cfg_gait['feet_link_names'])
cfg['robot']['link_names'] = cfg_gait['link_names']
cfg['robot']['dof_names'] = cfg_gait['dof_names']
cfg['robot']['diameter'] = cfg_gait['diameter']
cfg['robot']['mass'] = cfg_gait['mass']

if 'pose' not in cfg.keys():
    cfg['pose'] = {}
cfg['pose']['base_init_pos'] = cfg_gait['base_init_pos']
cfg['pose']['base_init_quat'] = cfg_gait['base_init_quat']
cfg['pose']['body_init_pos'] = cfg_gait['body_init_pos']
cfg['pose']['body_init_quat'] = cfg_gait['body_init_quat']
cfg['pose']['default_joint_angles'] = cfg_gait['default_joint_angles']
cfg['pose']['gait'] = cfg_gait['gait']

# cfg['pose']['base_init_pos'][2] += 0.05
# cfg['pose']['body_init_pos'][2] += 0.05

cfg['learning']['note'] = cfg_gait['task']

yaml.safe_dump(cfg, open(f"./cfgs/{args.robot}/{args.name}_final.yaml", "w"), sort_keys=False, default_flow_style=False)

### Build final training cfg

cfg_pose = cfg
cfg_train = yaml.safe_load(open(f"../cfgs/template.yaml"))

env_cfg = cfg_train['environment']

env_cfg['urdf_path'] = cfg_pose['robot']['asset_path']
env_cfg['robot_scale'] = cfg_pose['robot']['scale']
env_cfg['links_to_keep'] = cfg_pose['robot']['links_to_keep']
num_dof = len(cfg_pose['pose']['default_joint_angles'])
env_cfg['num_actions'] = num_dof
env_cfg['num_dofs'] = num_dof
env_cfg['num_states'] = 10 + num_dof * 2

terminate = []
for link in cfg_pose['robot']['link_names']:
    if link not in cfg_pose['robot']['foot_names']:
        terminate.append(link)
# import pdb; pdb.set_trace()
env_cfg['termination_contact_link_names'] = [cfg_pose['robot']['body_name'],]
env_cfg['penalized_contact_link_names'] = terminate.copy()
env_cfg['feet_link_names'] = cfg_pose['robot']['foot_names']
env_cfg['base_link_name'] = [cfg_pose['robot']['body_name']]

if type(cfg_pose['control']['kp']) != dict :
    cfg_pose['control']['kp'] = {'': cfg_pose['control']['kp']}
env_cfg['PD_stiffness'] = cfg_pose['control']['kp']
if type(cfg_pose['control']['kd']) != dict :
    cfg_pose['control']['kd'] = {'': cfg_pose['control']['kd']}
env_cfg['PD_damping'] = cfg_pose['control']['kd']
env_cfg['armature'] = cfg_pose['control']['armature']
env_cfg['dof_damping'] = cfg_pose['control']['damping'] 
env_cfg['dof_names'] = list(cfg_pose['pose']['default_joint_angles'].keys())
env_cfg['base_init_pos'] = cfg_pose['pose']['base_init_pos']
env_cfg['base_init_quat'] = cfg_pose['pose']['base_init_quat']
env_cfg['body_init_pos'] = cfg_pose['pose']['body_init_pos']
env_cfg['body_init_quat'] = cfg_pose['pose']['body_init_quat']
env_cfg['default_joint_angles'] = cfg_pose['pose']['default_joint_angles']

if 'base_reset_pos' in cfg_pose['pose'].keys():
    env_cfg['base_reset_pos'] = cfg_pose['pose']['base_reset_pos']
if 'base_reset_quat' in cfg_pose['pose'].keys():
    env_cfg['base_reset_quat'] = cfg_pose['pose']['base_reset_quat']
if 'reset_joint_angles' in cfg_pose['pose'].keys():
    env_cfg['reset_joint_angles'] = cfg_pose['pose']['reset_joint_angles']

num_feet = len(cfg_pose['robot']['foot_names'])
env_cfg['gait'] = cfg_pose['pose']['gait']
env_cfg['gait']['feet_height_target'] = [env_cfg['gait']['base_height_target'] * 0.4,] * num_feet

env_cfg['observation']['num_obs'] = 9 + num_dof * 3 + num_feet
env_cfg['observation']['num_priv_obs'] = 12 + num_dof * 4 + num_feet

env_cfg['command'] = cfg_pose['learning']['command']

cfg_train['reward_tuning']['num_iterations'] = cfg_pose['learning']['num_iterations']
cfg_train['reward_tuning']['num_samples'] = cfg_pose['learning']['num_samples']
cfg_train['reward_tuning']['num_logpoints'] = cfg_pose['learning']['num_logpoints']
cfg_train['reward_tuning']['note'] = cfg_pose['learning']['note']

yaml.safe_dump(cfg_train, 
                open(f"../cfgs/{args.robot}-{args.name}.yaml", "w"), 
                sort_keys=False,
                default_flow_style=False)








