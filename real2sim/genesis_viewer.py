import yaml

import genesis as gs
import mujoco as mj
import numpy as np
from genesis.utils import geom as gu
from config.rb_config import *

class GenesisViewer:
    def __init__(self, visualize=True, offset_config_path=None):
        # Genesis initialization
        gs.init(backend=gs.gpu)

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res=(1920, 1080),
                camera_pos=(3.5, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                refresh_rate=60,
                max_FPS=None,
            ),
            sim_options=gs.options.SimOptions(
                gravity=(0.0, 0.0, 0.0),
            ),
            show_viewer=visualize,
            show_FPS=False,
        )

        self.plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        self.robot = None
        self.robot_dofs = None

        self.rigid_bodies = {}
        self.rigid_body_offsets = {}
        self.target = {}

        # Only camera related
        self.world_rotation = np.eye(3)
        self.cameras = []

        if offset_config_path is not None:
            with open(offset_config_path, "r") as f:
                off = yaml.safe_load(f)
            for name, data in off.items():
                self.rigid_body_offsets[name] = {
                    "pos": np.array(data["pos"]),
                    "quat": np.array(data["quat"]),
                }
        self.offset_sampled = 0

    def load_rigid_body_by_name(self, name, mode="Sphere", params={}):
        rigid_body = None
        if mode == "Sphere":
            rigid_body = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=params.get("radius", 0.05),
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "Box":
            rigid_body = self.scene.add_entity(
                gs.morphs.Box(
                    size=params.get("size", (0.1, 0.1, 0.1)),
                    pos=(0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "File":
            rigid_body = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=params.get("path"),
                    scale=1.0,
                    pos=(0.0, 0.0, 0.0),
                    quat=(1.0, 0.0, 0.0, 0.0),
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Metal(
                    color=params.get("color", (1.0, 1.0, 1.0)),
                ),
            )
        elif mode == "Virtual": # No explicit build, only for calculation
            rigid_body = {
                "pos": np.array([0.0, 0.0, 0.0]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0])
            }
        else:
            raise NotImplementedError(f"Rigid mode {mode} not implemented!")
        self.rigid_bodies[name] = rigid_body

        if self.rigid_body_offsets.get(name) is None:
            self.rigid_body_offsets[name] = {
                "pos": np.array([0.0, 0.0, 0.0]),
                "quat": np.array([1.0, 0.0, 0.0, 0.0]),
            }

    def initialize_cameras(self, camera_calibrations, rotation_matrix):
        self.world_rotation = np.array(rotation_matrix)
        for cam in camera_calibrations:
            pos = self.world_rotation @ np.array(cam["Position"])
            cam_wxyz = np.roll(np.array(cam["Orientation"]), 1)
            quat = gu.R_to_quat(self.world_rotation @ gu.quat_to_R(cam_wxyz))
            camera = self.scene.add_entity(
                gs.morphs.Box(
                    size=(0.1, 0.1, 0.15),
                    pos=pos,
                    quat=quat,
                    collision=False,
                    fixed=True,
                ),
                surface=gs.surfaces.Plastic(
                    color=(0.5, 0.0, 1.0),
                ),
            )
            self.cameras.append(camera)
    
    def initialize_robot(self, mujoco_model=None):
        '''
        If mujoco_model is not None, the robot will always reorder the dofs according to the mujoco_model.
        '''
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="./assets/robot/unitree_g1/g1_custom_collision_29dof.urdf", # TODO
                pos=(0, 0, 0),
                scale=1.0,
                collision=False,
                merge_fixed_links=False,
            )
        )
        if mujoco_model is None:
            self.robot_dofs = list(range(6, 35))
        else:
            dof_names = [
                mj.mj_id2name(mujoco_model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(mujoco_model.njnt)
            ]
            self.robot_dofs = [
                self.robot.get_joint(name).dofs_idx_local[0] for name in dof_names[1:]
            ]
        return self.robot
    
    def build(self):
        self.scene.build()
    
    def update_dof_pos(self, dof_pos):
        '''
        Update the robot dof positions.
        3: pos, 4: quat, 29: dofs
        '''
        if self.robot is None:
            return
        self.robot.set_pos(dof_pos[:3])
        self.robot.set_quat(dof_pos[3:7])
        self.robot.set_dofs_position(dof_pos[7:], dofs_idx_local=self.robot_dofs)
    
    def ik_dof_pos(self):
        '''
        Compute the inverse kinematics for the robot.
        '''
        if self.robot is None:
            return
        for name, ik_joints in G1_IK_TABLES.items():
            link = self.robot.get_link(name)
            pos = self._get_pos_by_name(name)
            quat = self._get_quat_by_name(name)
            if ik_joints[0] == "pelvis":
                self.robot.set_pos(pos)
                self.robot.set_quat(quat)
            else:
                ik_masks = [self.robot.get_joint(joint).dofs_idx_local[0] for joint in ik_joints]
                qpos = self.robot.inverse_kinematics(
                    link=link,
                    pos=pos,
                    quat=quat,
                    dofs_idx_local=ik_masks
                )
                self.update_dof_pos(qpos)

    def update_rigid_body_by_name(self, name, pos, quat):
        '''
        Update a single rigid body by name.
        '''
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        aligned_quat, aligned_pos = self._transform_RT_by(
            quat, pos,
            self.rigid_body_offsets[name]["quat"], self.rigid_body_offsets[name]["pos"],
        )
        self._set_pos_by_name(name, aligned_pos)
        self._set_quat_by_name(name, aligned_quat)

    def update_rigid_bodies(self, frame):
        '''
        Update multiple rigid bodies according to mocap frame.
        '''
        for name, (pos, quat) in frame.items():
            if name in self.rigid_bodies:
                self.update_rigid_body_by_name(name, pos, quat)

    def update_rigid_body_offset_by_name(self, name, pos, quat, with_momentum=True):
        '''
        Update the offset for a single rigid body by name.
        Momentum m=1-1/n. n should not be 0 if with_momentum is True.
        '''
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        old_pos = self.rigid_body_offsets[name]["pos"]
        old_quat = self.rigid_body_offsets[name]["quat"]
        m = 1 - 1 / self.offset_sampled if with_momentum else 0
        blended_pos = m * old_pos + (1 - m) * pos
        blended_quat = m * old_quat + (1 - m) * quat
        blended_quat /= np.linalg.norm(blended_quat)  # nlerp works fine after 1st step

        self.rigid_body_offsets[name] = {
            "pos": blended_pos,
            "quat": blended_quat,
        }

    def _get_pos_by_name(self, name):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            return self.rigid_bodies[name].get_pos().cpu().numpy()
        elif isinstance(self.rigid_bodies[name], dict):
            return self.rigid_bodies[name]["pos"]
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")
    
    def _get_quat_by_name(self, name):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            return self.rigid_bodies[name].get_quat().cpu().numpy()
        elif isinstance(self.rigid_bodies[name], dict):
            return self.rigid_bodies[name]["quat"]
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")

    def _get_link_pos_by_name(self, name):
        if self.robot is None:
            raise ValueError("Robot not initialized!")
        link = self.robot.get_link(name)
        if link is None:
            raise ValueError(f"Link {name} not found in robot!")
        return link.get_pos().cpu().numpy()
    
    def _get_link_quat_by_name(self, name):
        if self.robot is None:
            raise ValueError("Robot not initialized!")
        link = self.robot.get_link(name)
        if link is None:
            raise ValueError(f"Link {name} not found in robot!")
        return link.get_quat().cpu().numpy()

    def _set_pos_by_name(self, name, pos):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            self.rigid_bodies[name].set_pos(pos)
        elif isinstance(self.rigid_bodies[name], dict):
            self.rigid_bodies[name]["pos"] = pos
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")

    def _set_quat_by_name(self, name, quat):
        if name not in self.rigid_bodies:
            raise ValueError(f"Rigid body {name} not found!")
        if isinstance(self.rigid_bodies[name], gs.engine.entities.RigidEntity):
            self.rigid_bodies[name].set_quat(quat)
        elif isinstance(self.rigid_bodies[name], dict):
            self.rigid_bodies[name]["quat"] = quat
        else:
            raise NotImplementedError(f"Unknown type for rigid body {name}!")
    
    def _transform_RT_by(self, R1, T1, R2, T2):
        '''
        Apply the offset (R2, T2) to the pose (R1, T1).
        R = R2 * R1
        T = R1 * T2 + T1
        '''
        R_out = gu.transform_quat_by_quat(R1, R2)
        T_out = gu.transform_by_quat(T2, R1) + T1
        return R_out, T_out

    def _get_RT_between(self, R1, T1, R2, T2):
        '''
        Get the offset from (R1, T1) to (R2, T2).
        R = R2 * R1^T
        T = R1^T * (T2 - T1)
        '''
        R_out = gu.transform_quat_by_quat(gu.inv_quat(R1), R2)
        T_out = gu.transform_by_quat(T2 - T1, gu.inv_quat(R1))
        return R_out, T_out

    def _get_offset_by_world(self, frame, target_link_name):
        '''
        Assume the target_link_name is accurately put at (trans, rot) in the world,
        calculate the offset for target_link_name under zero pose.
        '''
        if target_link_name not in frame.keys():
            raise ValueError(f"Target link {target_link_name} not found in frame!")

        R_m, T_m = np.array(frame[target_link_name][1]), np.array(frame[target_link_name][0])
        R_s, T_s = self._get_link_quat_by_name(target_link_name), self._get_link_pos_by_name(target_link_name)
        R_o, T_o = self._get_RT_between(R_m, T_m, R_s, T_s)
        return R_o, T_o

    def _get_offset_by_fixed_link(self, frame, fixed_link_name, target_link_name, joint_qpos):
        '''
        Assume the fixed_link_name has an accurate offset,
        calculate the offset for target_link_name under given joint_qpos.
        '''
        if fixed_link_name not in frame.keys() or target_link_name not in frame.keys():
            raise ValueError(f"Fixed link {fixed_link_name} or target link {target_link_name} not found in frame!")

        R_m1, T_m1 = np.array(frame[fixed_link_name][1]), np.array(frame[fixed_link_name][0])
        R_m2, T_m2 = np.array(frame[target_link_name][1]), np.array(frame[target_link_name][0])
        R_o1, T_o1 = self.rigid_body_offsets[fixed_link_name]["quat"], self.rigid_body_offsets[fixed_link_name]["pos"]
        R_s1, T_s1 = self._get_link_quat_by_name(fixed_link_name), self._get_link_pos_by_name(fixed_link_name)
        R_s2, T_s2 = self._get_link_quat_by_name(target_link_name), self._get_link_pos_by_name(target_link_name)
        R_oq, T_oq = self._get_RT_between(R_s1, T_s1, R_s2, T_s2)
        print(R_oq, T_oq)
        R_k1, T_k1 = self._transform_RT_by(R_m1, T_m1, R_o1, T_o1)
        R_k2, T_k2 = self._transform_RT_by(R_k1, T_k1, R_oq, T_oq)
        R_o2, T_o2 = self._get_RT_between(R_m2, T_m2, R_k2, T_k2)
        return R_o2, T_o2

    def calibrate_by_world(self, frame, trans=G1_CB1_POS, rot=G1_CB1_QUAT):
        self.offset_sampled += 1
        self.update_dof_pos(np.concatenate((trans, rot, np.zeros(29))))

        for name in G1_CB1_LINK_NAMES:
            R_o, T_o = self._get_offset_by_world(frame, name)
            self.update_rigid_body_offset_by_name(name, T_o, R_o)
    
    def calibrate_by_fixed_link(self, frame, joint_qpos):
        self.offset_sampled += 1
        self.update_dof_pos(np.concatenate((G1_CB1_POS, G1_CB1_QUAT, joint_qpos)))

        for name in G1_CB2_LINK_NAMES:
            for fixed_name in G1_CB1_LINK_NAMES:
                R_o, T_o = self._get_offset_by_fixed_link(frame, fixed_name, name, joint_qpos)
                self.update_rigid_body_offset_by_name(name, T_o, R_o)

    def save_offsets(self, save_path):
        save_data = {}
        def represent_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        yaml.add_representer(list, represent_list)
        for name, offset in self.rigid_body_offsets.items():
            save_data[name] = {
                "pos": offset["pos"].tolist(),
                "quat": offset["quat"].tolist(),
            }
        with open(save_path, "w") as f:
            yaml.dump(save_data, f, sort_keys=False)
        print(f"Offsets saved to {save_path}.")

    def step(self):
        self.scene.visualizer.update()
    
    ''' Setups '''
    
    def g1_links_setup_virtual(self):
        for name in G1_TRACKED_LINK_NAMES:
            self.load_rigid_body_by_name(
                name,
                mode="Virtual",
            )

    def g1_links_setup_offset(self):
        for name in G1_TRACKED_LINK_NAMES:
            self.load_rigid_body_by_name(
                name,
                mode="File",
                params={"path": f"./assets/robot/unitree_g1/meshes/{name}.STL", "color": (0.8, 0.0, 0.0)},
            )

    def MoCap_setup(self, args, mujoco_model=None):
        self.initialize_robot(mujoco_model=mujoco_model)
        self.build()
    
    def MoCap_step(self, qpos):
        self.update_dof_pos(qpos)
        self.step()
    
    def Real2Sim_setup(self, args):
        self.g1_links_setup_virtual()
        self.initialize_robot(mujoco_model=None)
        self.build()
    
    def Real2Sim_step(self, frame):
        self.update_rigid_bodies(frame)
        self.ik_dof_pos()
        self.step()
    
    def Real2Sim_offset_setup(self, args):
        self.g1_links_setup_offset()
        self.initialize_robot(mujoco_model=None)
        for name in G1_TRACKED_LINK_NAMES:
            self.target[name] = self.robot.get_link(name)
        self.build()

    def Real2Sim_offset_step(self, frame):
        self.update_rigid_bodies(frame)
        # offset = self._get_offsets(frame)
        self.step()
        return None