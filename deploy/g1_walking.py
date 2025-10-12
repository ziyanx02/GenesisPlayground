import time
from pathlib import Path
from typing import Union

import fire
import numpy as np
import torch
import sys

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import yaml_to_config  # type: ignore


def load_checkpoint_and_env_args(exp_name: str, num_ckpt: int | None = None):
    """Load JIT checkpoint and env_args from deploy/logs directory.
    
    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads the latest checkpoint.
        
    Returns:
        Tuple of (checkpoint_path, env_args)
    """
    from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs
    
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")
    
    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    
    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, LeggedRobotEnvArgs)
    
    # Load checkpoint
    if num_ckpt is not None:
        ckpt_path = deploy_dir / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # Find latest checkpoint
        ckpts = list(deploy_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {deploy_dir}")
        ckpt_path = max(ckpts, key=lambda p: int(p.stem.split("_")[-1]))
    
    print(f"Loading checkpoint from: {ckpt_path}")
    return ckpt_path, env_args


def to_numpy(data):
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    return np.array(data)


def env_args_to_handler_cfg(env_args):
    """Convert env_args (LeggedRobotEnvArgs) to handler cfg format.
    
    Args:
        env_args: LeggedRobotEnvArgs object
        
    Returns:
        Dictionary in the format expected by the handler
    """
    # Extract robot name from robot_args
    robot_name = env_args.robot_args.robot_name
    dof_names = env_args.robot_args.dof_names
    
    # Get default DOF positions
    default_dof_pos = {}
    for i, name in enumerate(dof_names):
        default_dof_pos[name] = float(env_args.robot_args.default_dof_pos[i])
    
    # Create cfg dict in the format expected by handler
    cfg = {
        "robot": {
            "name": robot_name,
        },
        "control": {
            "dof_names": dof_names,
            "kp": env_args.robot_args.kp,  # Assume this exists in robot_args
            "kd": env_args.robot_args.kd,  # Assume this exists in robot_args
            "default_dof_pos": default_dof_pos,
        }
    }
    
    return cfg


def main(
    exp_name: str = "walk",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = False,
    sim: bool = True,
    num_envs: int = 1,
):
    """Run policy on either simulation or real robot.
    
    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
        num_envs: Number of environments (only for sim mode)
    """
    device = "cpu" if not torch.cuda.is_available() else device
    
    # Load checkpoint and env_args
    ckpt_path, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt)
    
    # Load policy
    policy = torch.jit.load(str(ckpt_path))
    policy.to(device)
    policy.eval()
    
    # Create environment or handler
    env_idx = 0  # Initialize for both modes
    
    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as gs_envs
        
        envclass = getattr(gs_envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=num_envs,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        
        # Get default dof positions from robot
        default_dof_pos = to_numpy(env.robot.default_dof_pos)
        
        # For sim, reset_pos is the same as default
        reset_dof_pos = default_dof_pos.copy()
        
    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLowStateCmdHandler
        
        # Convert env_args to handler cfg format
        handler_cfg = env_args_to_handler_cfg(env_args)
        
        env = UnitreeLowStateCmdHandler(handler_cfg)
        env.init()
        env.start()
        
        # Wait for handler to be ready
        print("Waiting for robot to be ready...")
        while not env.Start:
            time.sleep(0.1)
        
        default_dof_pos = env.default_pos
        # Note: Assuming handler will have reset_pos property for safety ramping
        # If not available, use default_pos
        reset_dof_pos = getattr(env, 'reset_pos', env.default_pos).copy()
    
    # Initialize tracking variables
    last_action = np.zeros(len(env_args.robot_args.dof_names))
    commands = np.array([0.0, 0.0, 0.0])
    
    print("=" * 80)
    print("Starting policy execution")
    print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
    print(f"Policy: {ckpt_path}")
    print(f"Device: {device}")
    print("=" * 80)
    
    last_update_time = time.time()
    step_id = 0
    
    try:
        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, 'emergency_stop') and env.emergency_stop:  # type: ignore
                print("Emergency stop triggered!")
                break
            
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()
            
            # Update commands (can be modified for different behaviors)
            commands[0] = 1.0  # forward velocity (m/s)
            commands[1] = 0.0  # lateral velocity (m/s)
            commands[2] = 0.0  # angular velocity (rad/s)
            
            # Get observations from environment (unified interface)
            # Note: Handler should provide dof_pos, dof_vel, base_ang_vel properties
            # that match the sim environment interface
            if sim:
                # For sim, extract from the specific environment index
                dof_pos = to_numpy(env.dof_pos[env_idx])
                dof_vel = to_numpy(env.dof_vel[env_idx])
                projected_gravity = to_numpy(env.projected_gravity[env_idx])
                base_ang_vel = to_numpy(env.base_ang_vel[env_idx])
            else:
                # For real robot, assuming handler provides compatible interface
                # Handler should have properties: dof_pos, dof_vel, projected_gravity, base_ang_vel
                dof_pos = to_numpy(env.dof_pos)
                dof_vel = to_numpy(env.dof_vel)
                projected_gravity = to_numpy(env.projected_gravity)
                base_ang_vel = to_numpy(env.base_ang_vel)

            # Construct observation (matching training observation structure)
            obs = np.concatenate([
                last_action,
                (dof_pos - default_dof_pos) * 1.0,  # dof_pos offset
                dof_vel * env_args.obs_scales["dof_vel"],
                projected_gravity,
                base_ang_vel * env_args.obs_scales["base_ang_vel"],
                commands[:3],
            ])

            # Get action from policy
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t).squeeze(0).detach().cpu().numpy().astype(np.float32)

            # Apply action
            if sim:
                # For sim, convert action to torch and apply
                action_tensor = torch.from_numpy(action).to(torch.device(device)).unsqueeze(0)
                env.apply_action(action_tensor)
            else:
                # For real robot, apply with gradual interpolation for safety
                target_pos = reset_dof_pos + 0.3 * (
                    default_dof_pos + action * env_args.robot_args.action_scale - reset_dof_pos
                )
                if hasattr(env, 'target_pos'):
                    env.target_pos = target_pos  # type: ignore
                else:
                    print("Warning: Handler does not have target_pos attribute")
            
            last_action = action
            step_id += 1
            
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if not sim:
            print("Stopping robot handler...")
            # Handler cleanup if needed
        else:
            print("Simulation stopped.")
        print(f"Total steps: {step_id}")


if __name__ == "__main__":
    fire.Fire(main)
