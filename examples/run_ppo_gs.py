#!/usr/bin/env python3
"""Example: Train PPO on Genesis Goal-reaching environment using Genesis RL."""
import time
from pathlib import Path

import fire
import torch
from gs_agent.algos.ppo import PPO
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_env.sim.envs.manipulation.goal_reaching_env import GoalReachingEnv
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_agent.configs import PPO_DEFAULT, RUNNER_DEFAULT

def create_gs_env(
    env_name: str = "goal_reach_default", show_viewer: bool = False
) -> GenesisEnvWrapper:
    """Create gym environment wrapper."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GoalReachingEnv(args=EnvArgsRegistry[env_name], num_envs=1024, show_viewer=show_viewer, device=device)
    return GenesisEnvWrapper(env, device=device)


def create_ppo_runner_from_registry() -> OnPolicyRunner:
    """Create PPO runner using configuration from the registry."""
    # Environment setup
    wrapped_env = create_gs_env()

    # Create PPO algorithm
    ppo = PPO(
        env=wrapped_env,
        cfg=PPO_DEFAULT,
        device=wrapped_env.device,
    )

    # Create PPO runner
    runner = OnPolicyRunner(
        algorithm=ppo,
        runner_args=RUNNER_DEFAULT,
        device=wrapped_env.device,
    )
    return runner


def main(train: bool = True) -> None:
    """Main function demonstrating proper registry usage."""
    # Get configuration and runner from registry
    runner = create_ppo_runner_from_registry()
    # Set up logging with proper configuration
    logger = logger_configure(folder=str(runner.save_dir), format_strings=["stdout", "csv", "wandb"])
    # Train using Runner
    train_summary_info = runner.train(metric_logger=logger) 

    print("Training completed successfully!")
    print(f"Training completed in {train_summary_info['total_time']:.2f} seconds.")
    print(f"Total episodes: {train_summary_info['total_episodes']}.")
    print(f"Total steps: {train_summary_info['total_steps']}.")
    print(f"Total reward: {train_summary_info['final_reward']:.2f}.")

if __name__ == "__main__":
    fire.Fire(main)