#!/usr/bin/env python3
"""Example: Train PPO on WUJI Hand Retargeting task with object manipulation using Genesis RL.

This script trains a policy to follow reference hand trajectories from demonstrations
while manipulating objects, following the ManipTrans DexHandImitator architecture.
"""

import glob
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import gs_env.sim.envs as gs_envs
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_agent.algos.config.registry import PPO_HAND_IMITATOR_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_SINGLE_HAND_RETARGETING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from utils import apply_overrides_generic, config_to_yaml, plot_metric_on_axis


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> gs_envs.SingleHandRetargetingEnv:
    """Create gym environment wrapper with optional config overrides."""
    if torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    env_class = getattr(gs_envs, args.env_name)

    return env_class(
        args=args,
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
        eval_mode=eval_mode,
    )


def create_ppo_runner_from_registry(
    env: gs_envs.SingleHandRetargetingEnv,
    exp_name: str | None = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )

    # Create PPO runner
    if exp_name is not None:
        # Avoid mutating a frozen Pydantic model; create a copied config with updated save_path
        runner_args = runner_args.model_copy(update={"save_path": Path(f"./logs/{exp_name}")})
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=runner_args,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(
    exp_name: str,
    show_viewer: bool = False,
    num_ckpt: int | None = None,
    device: str = "cuda",
    env_args: Any = None,
    algo_cfg: Any = None,
) -> None:
    """Evaluate the policy."""
    print("=" * 80)
    print("EVALUATION MODE: Disabling domain randomization and observation noise")
    print("=" * 80)

    # Disable observation noise for evaluation
    env_args = env_args.model_copy(
        update={
            "obs_noises": {},  # Disable observation noise
        }
    )

    # Disable domain randomization for evaluation
    from gs_env.sim.robots.config.schema import DomainRandomizationArgs

    robot_args = env_args.robot_args.model_copy(
        update={
            "dr_args": DomainRandomizationArgs(
                kp_range=(1.0, 1.0),
                kd_range=(1.0, 1.0),
                motor_strength_range=(1.0, 1.0),
                motor_offset_range=(0.0, 0.0),
                friction_range=(1.0, 1.0),
                mass_range=(0.0, 0.0),
                com_displacement_range=(0.0, 0.0),
            )
        }
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Find the experiment directory without creating a new runner
    log_pattern = f"logs/{exp_name}/*"
    log_dirs = glob.glob(log_pattern)
    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    exp_dir = log_dirs[0]
    print(f"Loading policy from experiment: {exp_dir}")

    # Load checkpoint - either specific one or latest
    if num_ckpt is not None:
        ckpt_path = Path(exp_dir) / "checkpoints" / f"checkpoint_{num_ckpt:04d}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    else:
        ckpt_path = load_latest_model(Path(exp_dir))
        num_ckpt = int(ckpt_path.stem.split("_")[-1])

    print(f"Loading checkpoint: {ckpt_path}")

    # Create environment for evaluation
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Setup GIF recording if not showing viewer
    gif_path = None
    if not show_viewer:
        # Create gif directory structure
        gif_dir = Path("./gif") / exp_name
        gif_dir.mkdir(parents=True, exist_ok=True)

        gif_path = gif_dir / f"{num_ckpt}.gif"
        print(f"Will save GIF to: {gif_path}")

        # Start rendering
        env.start_rendering()  # type: ignore

    # Create PPO algorithm and load checkpoint
    ppo = PPO(
        env=wrapped_env,
        cfg=algo_cfg,
        device=wrapped_env.device,
    )
    ppo.load(ckpt_path, load_optimizer=False)

    # Get inference policy
    inference_policy = ppo.get_inference_policy()

    print("Starting evaluation...")

    def evaluate() -> None:
        nonlocal wrapped_env, inference_policy, gif_path, show_viewer, num_ckpt, exp_name
        if show_viewer:
            print("Running endlessly (press Ctrl+C to stop)")
        else:
            print("Running until all environments are done")

        step_count = 0
        total_reward = 0.0

        # For tracking hand and object trajectory following
        wrist_pos_errors = []
        wrist_rot_errors = []
        object_pos_errors = []
        object_rot_errors = []
        trajectory_progress = []

        # Reset environment
        obs, _ = wrapped_env.get_observations()

        # Create a wrapper that always uses deterministic=True
        class DeterministicWrapper(torch.nn.Module):
            def __init__(self, policy: Any) -> None:
                super().__init__()
                self.policy = policy

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                action, _ = self.policy(obs, deterministic=True)
                return action

        # Wrap and trace the policy with deterministic=True baked in
        wrapped_policy = DeterministicWrapper(inference_policy)
        inference_policy = torch.jit.trace(wrapped_policy, obs)

        # Save the JIT-traced policy and env_args to deploy folder
        print("Saving JIT-traced policy and env_args for deployment...")

        # Create deploy directory structure
        deploy_dir = Path("./deploy/logs") / exp_name
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Save JIT-traced policy
        jit_policy_path = deploy_dir / f"checkpoint_{num_ckpt}.pt"
        torch.jit.save(inference_policy, str(jit_policy_path))
        print(f"JIT-traced policy saved to: {jit_policy_path}")

        # Save env_args as YAML
        env_args_path = deploy_dir / "env_args.yaml"
        config_to_yaml(env_args, env_args_path)
        print(f"Environment args saved to: {env_args_path}")

        while True:
            # Get action from policy
            with torch.no_grad():
                action = inference_policy(obs)  # type: ignore[misc]

            # Step environment
            obs, reward, terminated, truncated, info = wrapped_env.step(action)

            cur_idx = wrapped_env.env.progress_buf[0].cpu().item()
            cur_traj_length = wrapped_env.env.env_traj_lengths[0].cpu().item()

            # Track wrist errors
            target_wrist_pos = wrapped_env.env._traj_data["wrist_pos"][0, cur_idx].cpu()
            current_wrist_pos = wrapped_env.env.base_pos[0].cpu()
            wrist_pos_error = torch.norm(target_wrist_pos - current_wrist_pos).item()
            wrist_pos_errors.append(wrist_pos_error)

            target_wrist_quat = wrapped_env.env._traj_data["wrist_quat"][0, cur_idx].cpu()
            current_wrist_quat = wrapped_env.env.base_quat[0].cpu()
            # Simple rotation error: 1 - |dot product|
            wrist_rot_error = 1 - torch.abs(torch.sum(target_wrist_quat * current_wrist_quat)).item()
            wrist_rot_errors.append(wrist_rot_error)

            # Track object errors
            target_obj_pos = wrapped_env.env._traj_data["obj_pos"][0, cur_idx].cpu()
            current_obj_pos = wrapped_env.env.object_pos[0].cpu()
            obj_pos_error = torch.norm(target_obj_pos - current_obj_pos).item()
            object_pos_errors.append(obj_pos_error)

            target_obj_quat = wrapped_env.env._traj_data["obj_quat"][0, cur_idx].cpu()
            current_obj_quat = wrapped_env.env.object_quat[0].cpu()
            obj_rot_error = 1 - torch.abs(torch.sum(target_obj_quat * current_obj_quat)).item()
            object_rot_errors.append(obj_rot_error)

            trajectory_progress.append(cur_idx / cur_traj_length)

            # Accumulate reward
            total_reward += reward.item()
            step_count += 1

            # Print progress
            if step_count % 50 == 0:
                print(f"Step {step_count}, Total reward: {total_reward:.2f}")
                if wrist_pos_errors:
                    avg_wrist_pos_error = np.mean(wrist_pos_errors[-50:])
                    avg_wrist_rot_error = np.mean(wrist_rot_errors[-50:])
                    avg_obj_pos_error = np.mean(object_pos_errors[-50:])
                    avg_obj_rot_error = np.mean(object_rot_errors[-50:])
                    print(f"  Avg wrist pos error (last 50 steps): {avg_wrist_pos_error:.4f} m")
                    print(f"  Avg wrist rot error (last 50 steps): {avg_wrist_rot_error:.4f}")
                    print(f"  Avg object pos error (last 50 steps): {avg_obj_pos_error:.4f} m")
                    print(f"  Avg object rot error (last 50 steps): {avg_obj_rot_error:.4f}")
                if trajectory_progress:
                    print(f"  Trajectory progress: {trajectory_progress[-1]*100:.1f}%")

            # Check if all environments are done (for non-viewer mode)
            if not show_viewer:
                if terminated.item() or truncated.item() or step_count > 500:
                    print(f"Episode ended at step {step_count}, Total reward: {total_reward:.2f}")
                    break
            else:
                # For viewer mode, check termination conditions
                if terminated.item() or truncated.item():
                    print(f"Episode ended at step {step_count}, Total reward: {total_reward:.2f}")
                    obs, _ = wrapped_env.get_observations()
                    total_reward = 0.0

        # Plot trajectory following metrics if data available
        if wrist_pos_errors and not show_viewer:
            print("Plotting trajectory following statistics...")
            fig, axes = plt.subplots(5, 1, figsize=(12, 16))

            steps = np.arange(len(wrist_pos_errors))

            # Wrist position error over time
            plot_metric_on_axis(
                axes[0],
                steps,
                [wrist_pos_errors],
                ["Wrist Position Error"],
                "Error (m)",
                "Wrist Position Tracking Error",
            )

            # Wrist rotation error over time
            plot_metric_on_axis(
                axes[1],
                steps,
                [wrist_rot_errors],
                ["Wrist Rotation Error"],
                "Error",
                "Wrist Rotation Tracking Error",
            )

            # Object position error over time
            plot_metric_on_axis(
                axes[2],
                steps,
                [object_pos_errors],
                ["Object Position Error"],
                "Error (m)",
                "Object Position Tracking Error",
            )

            # Object rotation error over time
            plot_metric_on_axis(
                axes[3],
                steps,
                [object_rot_errors],
                ["Object Rotation Error"],
                "Error",
                "Object Rotation Tracking Error",
            )

            # Trajectory progress over time
            if trajectory_progress:
                plot_metric_on_axis(
                    axes[4],
                    steps,
                    [trajectory_progress],
                    ["Progress"],
                    "Progress (0-1)",
                    "Trajectory Progress",
                    xlabel="Step",
                )

            # Save plot
            plot_dir = Path("./gif") / exp_name if exp_name else Path("./gif") / "latest"
            plot_dir.mkdir(parents=True, exist_ok=True)

            plot_path = plot_dir / f"{num_ckpt}_tracking.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close(fig)
            print(f"Tracking metrics plot saved to: {plot_path}")

        # Stop rendering and save GIF if recording
        if not show_viewer and gif_path is not None:
            print("Stopping rendering and saving GIF...")
            env.stop_rendering(save_gif=True, gif_path=str(gif_path))  # type: ignore
            print(f"GIF saved to: {gif_path}")

        print(f"Evaluation of checkpoint {ckpt_path} completed successfully!")
        print("Final evaluation results:")
        print(f"Total steps: {step_count}")
        print(f"Final reward: {total_reward:.2f}")
        if wrist_pos_errors:
            avg_wrist_pos_error = np.mean(wrist_pos_errors)
            avg_wrist_rot_error = np.mean(wrist_rot_errors)
            avg_obj_pos_error = np.mean(object_pos_errors)
            avg_obj_rot_error = np.mean(object_rot_errors)
            print(f"Average wrist position error: {avg_wrist_pos_error:.4f} m")
            print(f"Average wrist rotation error: {avg_wrist_rot_error:.4f}")
            print(f"Average object position error: {avg_obj_pos_error:.4f} m")
            print(f"Average object rotation error: {avg_obj_rot_error:.4f}")
        if trajectory_progress:
            print(f"Final trajectory progress: {trajectory_progress[-1]*100:.1f}%")

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=evaluate).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            evaluate()
    except KeyboardInterrupt:
        pass


def train_policy(
    exp_name: str | None = None,
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    use_wandb: bool = True,
    env_args: Any = None,
    algo_cfg: Any = None,
    runner_args: Any = None,
    video_log_freq: int | None = None,
) -> None:
    """Train the policy using PPO."""

    # Create environment
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=num_envs,
        device=device,
        args=env_args,
    )

    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry(
        env,
        exp_name=exp_name,
        algo_cfg=algo_cfg,
        runner_args=runner_args,
    )

    # Enable video logging if requested
    if video_log_freq is not None:
        runner.algorithm.set_video_log_freq(video_log_freq, fps=20)

    # Set up logging with proper configuration
    if exp_name is not None:
        save_path = Path(f"./logs/{exp_name}")
    else:
        save_path = runner_args.save_path
    logger_folder = save_path / datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logger_configure(
        folder=str(logger_folder),
        format_strings=["stdout", "csv", "wandb"],
        entity=None,
        project=None,
        exp_name=exp_name,
        mode="online" if use_wandb and (not show_viewer) else "disabled",
    )

    # Save configuration files to YAML
    print("Saving configuration files to YAML...")
    config_dir = logger_folder / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_to_yaml(env_args, config_dir / "env_args.yaml")
    config_to_yaml(algo_cfg, config_dir / "algo_cfg.yaml")
    config_to_yaml(runner_args, config_dir / "runner_args.yaml")

    # Train using Runner
    print("Starting training...")

    def train() -> None:
        nonlocal runner, logger
        train_summary_info = runner.train(metric_logger=logger)
        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total iterations: {train_summary_info['total_iterations']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Total reward: {train_summary_info['final_reward']:.2f}.")

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=train).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            train()
    except KeyboardInterrupt:
        pass


def main(
    num_envs: int = 4096,
    show_viewer: bool = False,
    device: str = "cuda",
    eval: bool = False,
    exp_name: str | None = None,
    num_ckpt: int | None = None,
    use_wandb: bool = True,
    env_name: str = "single_hand_retargeting",
    video_log_freq: int | None = 100,
    **cfg_overrides: Any,
) -> None:
    """Entry point.

    You can override configs via dot-notation, e.g.:
    --reward_args.WristPositionTrackingReward.scale=0.2
    --reward_args.FingerJointPositionTrackingReward.scale=1.5
    --trajectory_path=path/to/your/trajectory/directory
    --object_id=your_object_id
    --algo.lr=1e-3
    --runner.total_iterations=5000
    --video_log_freq=100  # Log video every 100 iterations (set to None to disable)
    """
    # Bucket overrides into env / algo / runner
    env_overrides: dict[str, Any] = {}
    algo_overrides: dict[str, Any] = {}
    runner_overrides: dict[str, Any] = {}

    for k, v in cfg_overrides.items():
        if k.startswith("cfgs.env.") or k.startswith("env.") or k.startswith("reward_args."):
            env_overrides[k] = v
            continue
        if k.startswith("cfgs.algo.") or k.startswith("algo."):
            algo_overrides[k] = v
            continue
        if k.startswith("cfgs.runner.") or k.startswith("runner."):
            runner_overrides[k] = v
            continue

    env_args = EnvArgsRegistry[env_name]
    env_args = apply_overrides_generic(env_args, env_overrides, prefixes=("cfgs.", "env."))
    algo_cfg = apply_overrides_generic(PPO_HAND_IMITATOR_MLP, algo_overrides, prefixes=("cfgs.", "algo."))
    runner_args = apply_overrides_generic(
        RUNNER_SINGLE_HAND_RETARGETING_MLP, runner_overrides, prefixes=("cfgs.", "runner.")
    )

    if eval:
        # Evaluation mode - don't create runner to avoid creating empty log dir
        num_envs = 1
        print("Evaluation mode: Loading trained policy")
        evaluate_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_ckpt=num_ckpt,
            device=device,
            env_args=env_args,
            algo_cfg=algo_cfg,
        )
    else:
        # Training mode
        print("Training mode: Starting policy training")
        train_policy(
            exp_name=exp_name,
            show_viewer=show_viewer,
            num_envs=num_envs,
            device=device,
            use_wandb=use_wandb,
            env_args=env_args,
            algo_cfg=algo_cfg,
            runner_args=runner_args,
            video_log_freq=video_log_freq,
        )


if __name__ == "__main__":
    fire.Fire(main)
