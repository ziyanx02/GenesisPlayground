#!/usr/bin/env python3
"""Example: Train PPO on Genesis Walking environment using Genesis RL."""

import glob
import os
from pathlib import Path

import fire
import torch
from gs_agent.algos.config.registry import PPO_WALKING_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_WALKING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.common.bases.base_env import BaseEnv
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.locomotion.walking_env import WalkingEnv


def create_gs_env(
    env_name: str = "walk_default",
    show_viewer: bool = False,
    num_envs: int = 2048,
    device: str = "cuda",
) -> BaseEnv:
    """Create gym environment wrapper."""
    if torch.backends.mps.is_available():
        device_tensor = torch.device("mps")
    elif torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    return WalkingEnv(
        args=EnvArgsRegistry[env_name],
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
    )


def create_ppo_runner_from_registry(env: BaseEnv, exp_name: str | None = None) -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_WALKING_MLP,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner_args = RUNNER_WALKING_MLP
    if exp_name is not None:
        runner_args.save_path = Path(f"./logs/{exp_name}")
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=RUNNER_WALKING_MLP,
        device=wrapped_env.device,
    )
    return runner


def evaluate_policy(
    exp_name: str | None = None, show_viewer: bool = False, num_ckpt: int | None = None
) -> None:
    """Evaluate the policy."""
    # Find the experiment directory without creating a new runner
    if exp_name is not None:
        # Use specific experiment name
        log_pattern = f"logs/ppo_gs_{exp_name}/*"
        log_dirs = glob.glob(log_pattern)
        if not log_dirs:
            raise FileNotFoundError(
                f"No experiment directories found matching pattern: {log_pattern}"
            )
    else:
        # Find latest experiment - try walking first, then goal_reaching
        log_patterns = ["logs/ppo_gs_walking/*", "logs/ppo_gs_goal_reaching/*"]
        log_dirs = []
        for pattern in log_patterns:
            log_dirs.extend(glob.glob(pattern))
            if log_dirs:  # If we found any, use this pattern
                break

        if not log_dirs:
            raise FileNotFoundError(
                f"No experiment directories found. Tried patterns: {log_patterns}"
            )

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

    print(f"Loading checkpoint: {ckpt_path}")

    # Create environment for evaluation
    env = create_gs_env(show_viewer=show_viewer, num_envs=1, device="cuda")
    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create PPO algorithm and load checkpoint
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_WALKING_MLP,
        device=wrapped_env.device,
    )
    ppo.load(ckpt_path, load_optimizer=False)
    ppo.eval_mode()

    # Get inference policy
    inference_policy = ppo.get_inference_policy()

    # Reset environment
    obs = wrapped_env.get_observations()

    print("Starting evaluation...")
    if show_viewer:
        print("Running endlessly (press Ctrl+C to stop)")
    else:
        print("Running for 500 steps")

    step_count = 0
    total_reward = 0.0

    try:
        while True:
            # Get action from policy
            with torch.no_grad():
                action, _log_prob = inference_policy(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, _ = wrapped_env.step(action)

            # Accumulate reward
            total_reward += reward.item()
            step_count += 1

            # Print progress
            if step_count % 50 == 0:
                print(f"Step {step_count}, Total reward: {total_reward:.2f}")

            # Check termination conditions
            if not show_viewer and step_count >= 500:
                print(f"Evaluation completed after {step_count} steps")
                break

            # Handle episode resets
            if terminated.item() or truncated.item():
                print(f"Episode ended at step {step_count}, Total reward: {total_reward:.2f}")
                obs = wrapped_env.get_observations()
                total_reward = 0.0

    except KeyboardInterrupt:
        print(f"\nEvaluation interrupted at step {step_count}")

    print("Final evaluation results:")
    print(f"  Total steps: {step_count}")
    print(f"  Final reward: {total_reward:.2f}")


def main(
    num_envs: int = 2048,
    show_viewer: bool = False,
    device: str = "cuda",
    eval: bool = False,
    exp_name: str | None = None,
    num_ckpt: int | None = None,
) -> None:
    """Main function demonstrating proper registry usage."""
    if eval:
        # Evaluation mode - don't create runner to avoid creating empty log dir
        print("Evaluation mode: Loading trained policy")
        evaluate_policy(exp_name=exp_name, show_viewer=show_viewer, num_ckpt=num_ckpt)
    else:
        # Training mode
        # create environment
        env = create_gs_env(show_viewer=show_viewer, num_envs=num_envs, device=device)
        # Get configuration and runner from registry
        runner = create_ppo_runner_from_registry(env, exp_name=exp_name)
        # Set up logging with proper configuration
        logger = logger_configure(
            folder=str(runner.save_dir),
            format_strings=["stdout", "csv", "wandb"],
            entity=None,
            project=None,
            exp_name=exp_name,
            mode="online" if not show_viewer else "disabled",
        )
        # Train using Runner
        train_summary_info = runner.train(metric_logger=logger)

        print("Training completed successfully!")
        print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
        print(f"Total episodes: {train_summary_info['total_episodes']}.")
        print(f"Total steps: {train_summary_info['total_steps']}.")
        print(f"Total reward: {train_summary_info['final_reward']:.2f}.")


if __name__ == "__main__":
    fire.Fire(main)
