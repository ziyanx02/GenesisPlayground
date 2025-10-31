"""In-hand rotation environment for WUJI hand."""
import importlib
from typing import Any

import genesis as gs
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from gs_env.common.bases.base_env import BaseEnv
from gs_env.common.utils.math_utils import quat_to_euler
from gs_env.common.utils.misc_utils import get_space_dim
from gs_env.sim.envs.config.schema import ManipulationEnvArgs
from gs_env.sim.robots.manipulators import WUJIHand
from gs_env.sim.scenes import FlatScene

_DEFAULT_DEVICE = torch.device("cpu")


class InHandRotationEnv(BaseEnv):
    """
    In-hand rotation environment for WUJI dexterous hand.
    Task: Rotate a cube around its Z-axis using the hand.

    Observations:
    - Actor: Configurable observation terms (e.g., hand_dof_pos, hand_dof_vel, action history)
    - Critic: Actor observations + additional cube state information

    Actions (20D): Delta joint positions for 20 finger joints

    Rewards:
        - Cube Z-axis rotation velocity
        - Penalties for dropping the cube
    """

    def __init__(
        self,
        args: ManipulationEnvArgs,
        num_envs: int,
        show_viewer: bool = False,
        device: torch.device = _DEFAULT_DEVICE,
        eval_mode: bool = False,
    ) -> None:
        super().__init__(device=device)
        self._num_envs = num_envs
        self._device = device
        self._show_viewer = show_viewer
        self._args = args
        self._eval_mode = eval_mode

        if not gs._initialized:  # noqa: SLF001
            gs.init(performance_mode=True, backend=getattr(gs.constants.backend, device.type))

        # == setup the scene ==
        self._scene = FlatScene(
            num_envs=self._num_envs,
            args=args.scene_args,
            show_viewer=self._show_viewer,
            img_resolution=args.img_resolution,
        )

        # == setup the robot (WUJI hand) ==
        self._robot = WUJIHand(
            num_envs=self._num_envs,
            scene=self._scene.scene,
            args=args.robot_args,
            device=self.device,
        )
        self._action_dim = self._robot._arm_dof_dim + self._robot._gripper_dof_dim

        # == setup cube object ==
        self._cube = self._scene.scene.add_entity(
            gs.morphs.Box(
                size=(self._args.cube_args["size"],) * 3,
                pos=self._args.cube_args["position"],
            ),
            material=gs.materials.Rigid(
                rho=100.0,  # density
                friction=0.8,  # friction for grasping
            ),
        )

        # == set up camera for rendering ==
        self._floating_camera = self._scene.scene.add_camera(
            res=(480, 480),
            fov=60,
            GUI=False,
        )

        # == build the scene ==
        self._scene.build()

        # == initialize robot limits after scene is built ==
        self._robot.post_build_init(eval_mode=eval_mode)

        # == setup reward scalars and functions ==
        dt = self._scene.scene.dt
        self._reward_functions = {}
        self._reward_required_keys: set[str] = set()
        module_name = f"gs_env.common.rewards.{self._args.reward_term}_terms"
        module = importlib.import_module(module_name)
        for key in args.reward_args.keys():
            reward_cls = getattr(module, key, None)
            if reward_cls is None:
                raise ValueError(f"Reward {key} not found in rewards module.")
            scale = args.reward_args[key]["scale"] * dt
            other_args = {k: v for k, v in args.reward_args[key].items() if k != "scale"}
            reward_instance = reward_cls(scale=scale, **other_args)
            self._reward_functions[key] = reward_instance
            # Record declared inputs for this reward term
            if hasattr(reward_instance, "required_keys"):
                self._reward_required_keys.update(getattr(reward_instance, "required_keys", ()))

        # Environment parameters
        self._max_sim_time = 20.0  # seconds

        # Initialize buffers
        self._init()
        self.reset()

    def _init(self) -> None:
        """Initialize observation and action spaces and environment buffers."""

        # Action space: 20D for finger joints + 1D for gripper (we'll ignore gripper)
        self._action_space = self._robot.action_space  # TODO: is this useful?

        # Action history length
        self._action_history_len = self._args.obs_history_len
        self._action_latency = self._args.action_latency

        # Action and DOF position buffers with history
        self._action_buf = torch.zeros(
            (self.num_envs, self._action_dim, self._action_history_len + 1), device=self._device
        )
        self._dof_pos_buf = self._robot._default_dof_pos[None, :, None].repeat(
            self.num_envs, 1, self._action_history_len + 1
        )

        # Create flattened observation buffers (these will be updated in update_buffers)
        # These are needed for observation space initialization
        self.action_history_flat = torch.zeros(
            (self.num_envs, self._action_dim * self._action_history_len), device=self._device
        )
        self.dof_pos_history_flat = torch.zeros(
            (self.num_envs, self._action_dim * self._action_history_len), device=self._device
        )

        # Cube state buffers
        self.cube_pos = torch.zeros((self.num_envs, 3), device=self._device)
        self.cube_quat = torch.zeros((self.num_envs, 4), device=self._device)
        self.cube_lin_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.cube_ang_vel = torch.zeros((self.num_envs, 3), device=self._device)
        self.cube_euler = torch.zeros((self.num_envs, 3), device=self._device)

        # Hand state buffers
        self.hand_dof_pos = torch.zeros((self.num_envs, self._action_dim), device=self._device)
        self.hand_dof_vel = torch.zeros((self.num_envs, self._action_dim), device=self._device)

        # Hand palm position (for fall-off detection)
        self.hand_palm_pos = torch.zeros((self.num_envs, 3), device=self._device)

        # Environment state buffers
        self.time_since_reset = torch.zeros(self.num_envs, device=self._device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self._device)

        # Initial cube pose (in hand, above palm)
        self.initial_cube_height = 0.20  # 20cm above ground (hand palm is at ~15cm)

        # Action scale for delta control (from robot args)
        self._action_scale = self._args.robot_args.action_scale

        # Data used for reward calculation
        # Fingertip positions (flattened 15D: 5 fingertips × 3D each)
        self.fingertip_pos = torch.zeros((self.num_envs, 15), device=self._device)
        # Initial DOF positions (for pose difference penalty)
        self.init_dof_pos = self._robot._default_dof_pos.repeat(self.num_envs, 1).to(self._device)
        # Rotation axis buffer (for rotation rewards)
        self.rot_axis = torch.zeros((self.num_envs, 3), device=self._device)
        # Default to Z-axis rotation (like penspin with +z)
        self.rot_axis[:, 2] = 1.0
        # Torques buffer (for torque penalties) - approximated from actions
        self.torques = torch.zeros((self.num_envs, self._action_dim), device=self._device)
        # reference point for termination calculation
        self.stay_center = torch.tensor(self._args.reward_args["EarlyTerminationPenalty"]["stay_center"], device=self._device).unsqueeze(0)

        # Rendering buffers
        self._rendered_images = []
        self._rendering = False
        self.camera_lookat = torch.tensor([0.0, 0.0, 0.2], device=self._device)
        self.camera_pos = torch.tensor([0.3, -0.3, 0.3], device=self._device)

        # Build observation spaces dynamically from args
        actor_obs_spaces = {}
        for obs_term in self._args.actor_obs_terms:
            assert hasattr(self, obs_term), f"Observation term {obs_term} not found in the environment."
            actor_obs_spaces[obs_term] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(getattr(self, obs_term).shape[-1],),
                dtype=np.float32,
            )
        self._actor_observation_space = gym.spaces.Dict(actor_obs_spaces)

        critic_obs_spaces = {}
        for obs_term in self._args.critic_obs_terms:
            assert hasattr(self, obs_term), f"Observation term {obs_term} not found in the environment."
            critic_obs_spaces[obs_term] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(getattr(self, obs_term).shape[-1],),
                dtype=np.float32,
            )
        self._critic_observation_space = gym.spaces.Dict(critic_obs_spaces)
        self._info_space = gym.spaces.Dict({})
        self._extra_info = {}

    def reset_idx(self, envs_idx: torch.IntTensor) -> None:
        """Reset specified environments."""
        if len(envs_idx) == 0:
            return

        # Reset robot to default pose
        self._robot.reset(envs_idx=envs_idx)

        # Reset cube to initial position (in hand)
        num_reset = len(envs_idx)
        cube_init_pos = torch.tensor(self._args.cube_args["position"], device=self._device).unsqueeze(0).repeat(num_reset, 1)

        # Add small random offset if not in eval mode
        if not self._eval_mode:
            cube_init_pos[:, :2] += (torch.rand((num_reset, 2), device=self._device) - 0.5) * 0.02
            cube_init_pos[:, 2] += (torch.rand(num_reset, device=self._device) - 0.5) * 0.01

        # Random initial orientation
        cube_init_quat = torch.zeros((num_reset, 4), device=self._device)
        cube_init_quat[:, 0] = 1.0  # w=1, identity quaternion
        if not self._eval_mode:
            # Random rotation around Z axis
            random_yaw = (torch.rand(num_reset, device=self._device) - 0.5) * np.pi
            cube_init_quat[:, 0] = torch.cos(random_yaw / 2)
            cube_init_quat[:, 3] = torch.sin(random_yaw / 2)

        # Set cube pose
        cube_pose = torch.cat([cube_init_pos, cube_init_quat], dim=-1)
        self._cube.set_qpos(cube_pose, envs_idx=envs_idx, zero_velocity=True)

        # Reset buffers
        self.time_since_reset[envs_idx] = 0.0
        self._action_buf[envs_idx] = 0.0
        self._dof_pos_buf[envs_idx] = self._robot._default_dof_pos[None, :, None].repeat(
            len(envs_idx), 1, self._action_history_len + 1
        )
        # Reset torques for penspin-style reward calculation
        self.torques[envs_idx] = 0.0

    def get_terminated(self) -> torch.Tensor:
        """Check if episodes should terminate."""
        reset_buf = self.get_truncated()

        # Terminate if cube falls below threshold relative to hand palm
        cube_height_above_hand = self.cube_pos[:, 2] - self.stay_center[:, 2]
        cube_dropped = cube_height_above_hand < self._args.reward_args["EarlyTerminationPenalty"]["height_threshold"]
        reset_buf |= cube_dropped

        # Terminate if cube moves too far from hand in XY
        cube_xy_dist = torch.norm(self.cube_pos[:, :2] - self.stay_center[:, :2], dim=-1)
        cube_too_far = cube_xy_dist > self._args.reward_args["EarlyTerminationPenalty"]["xy_threshold"]
        reset_buf |= cube_too_far

        self.reset_buf[:] = reset_buf
        termination_dict = {}
        termination_dict["cube_dropped"] = cube_dropped.clone()
        termination_dict["cube_too_far"] = cube_too_far.clone()
        termination_dict["any"] = reset_buf.clone()
        self._extra_info["termination"] = termination_dict

        return reset_buf

    def get_truncated(self) -> torch.Tensor:
        """Check if episodes should truncate due to time limit."""
        if self._eval_mode:
            self._max_sim_time = float("inf")
        time_out_buf = self.time_since_reset > self._max_sim_time
        self.time_out_buf[:] = time_out_buf
        return time_out_buf

    def update_buffers(self) -> None:
        """Update all state buffers from simulator."""
        # Update cube state
        self.cube_pos[:] = self._cube.get_pos()
        self.cube_quat[:] = self._cube.get_quat()
        self.cube_euler[:] = quat_to_euler(self.cube_quat)

        self.cube_lin_vel[:] = self._cube.get_vel()
        self.cube_ang_vel[:] = self._cube.get_ang()

        # Update hand state (get joint positions and velocities)
        self.hand_dof_pos[:] = self._robot.dof_pos
        self.hand_dof_vel[:] = self._robot.dof_vel

        # Update hand palm position, which is just the root position
        self.hand_palm_pos[:] = self._robot.base_pos

        self.fingertip_pos[:] = self._robot.fingertip_pos.reshape(self.num_envs, -1)

        # Update flattened history buffers for observations
        self.action_history_flat[:] = self._action_buf[:, :, -self._action_history_len:].transpose(1, 2).reshape(
            self.num_envs, -1
        )
        self.dof_pos_history_flat[:] = self._dof_pos_buf[:, :, -self._action_history_len:].transpose(1, 2).reshape(
            self.num_envs, -1
        )

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get actor and critic observations."""
        self.update_buffers()

        # Build actor observation from configured terms
        obs_components = []
        for key in self._args.actor_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_noise = torch.randn_like(obs_gt) * self._args.obs_noises.get(key, 0.0)
            if self._eval_mode:
                obs_noise *= 0
            obs_components.append(obs_gt + obs_noise)
        actor_obs = torch.cat(obs_components, dim=-1)

        # Build critic observation from configured terms
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        critic_obs = torch.cat(obs_components, dim=-1)

        return actor_obs, critic_obs

    def _pre_step(self) -> None:
        """Update timers before each physics substep."""
        self.time_since_reset += self._scene.scene.dt

    def apply_action(self, action: torch.Tensor) -> None:
        """Apply action to the environment."""
        # Action is 20D delta for finger joints
        action = action.detach().to(self._device)

        # Update action history buffer (shift and add new action)
        self._action_buf[:] = torch.cat([self._action_buf[:, :, 1:], action.unsqueeze(-1)], dim=-1)

        # Compute target position: current_position + action_scale * delta_action
        # Get action from history buffer with latency
        exec_action = self._action_buf[:, :, -self._action_latency - 1]

        # Scale the action
        exec_action *= self._action_scale

        # Target position = current + scaled delta
        target_dof_pos = self.hand_dof_pos + exec_action

        self.torques *= 0
        # Apply actions and simulate physics with decimation
        for _ in range(self._args.robot_args.decimation):
            self._pre_step()

            self._robot.apply_action(action=target_dof_pos)
            self._scene.scene.step()
            self.torques = torch.max(self.torques, torch.abs(self._robot.torque))
        
        self.update_buffers()

        # Update DOF position history buffer (after all substeps)
        self._dof_pos_buf[:] = torch.cat(
            [self._dof_pos_buf[:, :, 1:], self.hand_dof_pos.unsqueeze(-1)], dim=-1
        )

        # Render if rendering is enabled
        self._render_headless()

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment."""
        # Apply action
        self.apply_action(action)

        # Get terminated
        terminated = self.get_terminated()
        if terminated.dim() == 1:
            terminated = terminated.unsqueeze(-1)

        # Get truncated
        truncated = self.get_truncated()
        if truncated.dim() == 1:
            truncated = truncated.unsqueeze(-1)

        # Get reward
        reward, reward_terms = self.get_reward()
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)

        # Update history
        self.update_history()

        # Get extra infos
        extra_infos = self.get_extra_infos()
        extra_infos["reward_terms"] = reward_terms

        # Reset if terminated or truncated
        done_idx = terminated.nonzero(as_tuple=True)[0]
        if len(done_idx) > 0:
            self.reset_idx(done_idx)

        # Get observations
        next_obs, _ = self.get_observations()

        return next_obs, reward, terminated, truncated, extra_infos

    def update_history(self) -> None:
        # Currently not used
        pass

    def get_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute rewards using additive composition (penspin-style)."""
        reward_total = torch.zeros(self.num_envs, device=self._device)
        reward_dict = {}

        # Prepare state dict for reward functions
        state_dict = {key: getattr(self, key) for key in self._reward_required_keys}

        # Compute all configured rewards and sum them
        # Each reward term already has its scale applied in the RewardTerm class
        for key, func in self._reward_functions.items():
            reward = func(state_dict)
            reward_total += reward
            reward_dict[f"{key}"] = reward.clone()

        reward_dict["Total"] = reward_total

        return reward_total, reward_dict

        # reward_total = torch.zeros(self.num_envs, device=self._device)
        # reward_total_pos = torch.zeros(self.num_envs, device=self._device)
        # reward_total_neg = torch.zeros(self.num_envs, device=self._device)
        # reward_dict = {}

        # state_dict = {key: getattr(self, key) for key in self._reward_required_keys}
        # for key, func in self._reward_functions.items():
        #     reward = func(state_dict)
        #     if reward.sum() >= 0:
        #         reward_total_pos += reward
        #     else:
        #         reward_total_neg += reward
        #     reward_dict[f"{key}"] = reward.clone()
        # reward_total = reward_total_pos * torch.exp(reward_total_neg)
        # reward_dict["Total"] = reward_total
        # reward_dict["TotalPositive"] = reward_total_pos
        # reward_dict["TotalNegative"] = reward_total_neg

        # return reward_total, reward_dict

    def get_extra_infos(self) -> dict[str, Any]:
        self.update_buffers()
        obs_components = []
        for key in self._args.critic_obs_terms:
            obs_gt = getattr(self, key) * self._args.obs_scales.get(key, 1.0)
            obs_components.append(obs_gt)
        obs_tensor = torch.cat(obs_components, dim=-1)
        self._extra_info["observations"] = {"critic": obs_tensor}
        self._extra_info["time_outs"] = self.time_out_buf.clone()[:, None]
        return self._extra_info

    @property
    def num_envs(self) -> int:
        return self._scene.num_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def actor_obs_dim(self) -> int:
        return get_space_dim(self._actor_observation_space)

    @property
    def critic_obs_dim(self) -> int:
        return get_space_dim(self._critic_observation_space)

    @property
    def scene(self) -> FlatScene:
        return self._scene

    @property
    def robot(self) -> WUJIHand:
        return self._robot

    @property
    def dt(self) -> float:
        """Environment timestep (accounts for decimation)."""
        return self._scene.scene.dt * self._args.robot_args.decimation

    def _render_headless(self) -> None:
        """Render a frame from the floating camera if rendering is enabled."""
        if self._rendering and len(self._rendered_images) < 1000:
            hand_pos = self.hand_palm_pos[0]
            self._floating_camera.set_pose(
                pos=hand_pos + self.camera_pos,
                lookat=hand_pos + self.camera_lookat,
            )
            rgb, _, _, _ = self._floating_camera.render()
            self._rendered_images.append(rgb)

    def start_rendering(self) -> None:
        """Start recording rendered images."""
        self._rendering = True
        self._rendered_images = []

    def stop_rendering(self, save_gif: bool = True, gif_path: str = ".") -> None:
        """Stop recording and optionally save as GIF."""
        self._rendering = False
        if save_gif and self._rendered_images:
            self.save_gif(gif_path)

    def save_gif(self, gif_path: str, duration: int = 100) -> None:
        """
        Save the rendered images as a GIF.

        Args:
            gif_path: Path to save the GIF.
            duration: Duration of each frame in milliseconds (default: 20ms)
        """
        if not self._rendered_images:
            print("No rendered images to save.")
            return

        # Convert numpy arrays to PIL Images
        pil_images = []
        for img_array in self._rendered_images:
            # Convert from numpy array to PIL Image
            # Assuming the image is in RGB format (H, W, 3)
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_images.append(pil_img)

        # Save as GIF
        if pil_images:
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration,
                loop=0,  # Infinite loop
            )
            print(f"GIF saved to: {gif_path}")
        else:
            print("No images to save as GIF.")

    def eval(self) -> None:
        """Set environment to evaluation mode."""
        self._eval_mode = True
