#!/usr/bin/env python3
"""Example: Train PPO on Genesis Walking environment using Genesis RL."""

import glob
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_agent.algos.config.registry import PPO_WALKING_MLP
from gs_agent.algos.ppo import PPO
from gs_agent.runners.config.registry import RUNNER_WALKING_MLP
from gs_agent.runners.onpolicy_runner import OnPolicyRunner
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_latest_model
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.robots.config.registry import DRArgsRegistry
from gs_env.sim.envs.locomotion.walking_env import WalkingEnv
from utils import apply_overrides_generic, plot_metric_on_axis


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    args: Any = None,
) -> WalkingEnv:
    """Create gym environment wrapper with optional config overrides."""
    if torch.cuda.is_available() and device == "cuda":
        device_tensor = torch.device("cuda")
    else:
        device_tensor = torch.device("cpu")
    print(f"Using device: {device_tensor}")

    return WalkingEnv(
        args=args,
        num_envs=num_envs,
        show_viewer=show_viewer,
        device=device_tensor,  # type: ignore
    )


def run_dof_diagnosis(
    env: GenesisEnvWrapper,
    dof_idx: int = 0,
    num_dofs: int = 29,
    wave_type: str = "SIN",
    period: int = 1,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Run DOF diagnosis."""
    print(f"Running DOF diagnosis for {wave_type}")
    NUM_TOTAL_PERIODS = 5
    linear_phase = lambda x: x / period * 2 * np.pi
    quadratic_phase = lambda x: (x / period) ** 2 * 2 * np.pi / NUM_TOTAL_PERIODS * 3
    linear_amp = lambda x: amplitude * x / (NUM_TOTAL_PERIODS * period)
    if wave_type == "SIN":
        wave_func = lambda x: np.sin(linear_phase(x)) * amplitude
    elif wave_type == "SQ":
        wave_func = lambda x: np.sign(np.sin(linear_phase(x))) * amplitude
    elif wave_type == "AFM-SIN":
        wave_func = lambda x: np.sin(quadratic_phase(x))  * linear_amp(x)
    elif wave_type == "AFM-SQ":
        wave_func = lambda x: np.sign(np.sin(quadratic_phase(x))) * linear_amp(x)
    else:
        raise ValueError(f"Invalid wave type: {wave_type}")
    
    action = torch.zeros((env.num_envs, num_dofs), device=env.device)
    action[:, dof_idx] = offset

    for _ in range(50): 
        env.step(action)

    target_dof_pos_list = []
    dof_pos_list = []

    for i in range(period * NUM_TOTAL_PERIODS): 
        target_dof_pos = wave_func(i) + offset
        action[:, dof_idx] = target_dof_pos / env.env.action_scale
        dof_pos = (env.env.dof_pos[0] - env.env.robot.default_dof_pos)[dof_idx].cpu().item()
        target_dof_pos_list.append(target_dof_pos)
        dof_pos_list.append(dof_pos)
        env.step(action)

        if i % 10 == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    return target_dof_pos_list, dof_pos_list


def main(
    show_viewer: bool = False,
    device: str = "cuda",
    env_name: str = "g1_fixed",
) -> None:

    env_args = EnvArgsRegistry[env_name]
    env_args = env_args.model_copy(
        update={
            "obs_noises": {},  # Disable observation noise
        }
    )
    robot_args = env_args.robot_args.model_copy(
        update={
            "dr_args": DRArgsRegistry["no_randomization"],
        }
    )
    env_args = env_args.model_copy(update={"robot_args": robot_args})

    # Create environment for evaluation
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
    )
    env.stop_random_push()

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # Create figure with 4 subplots in one column
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))

    for i, wave_type in enumerate(["SIN", "SQ", "AFM-SIN", "AFM-SQ"]):
        target_dof_pos_list, dof_pos_list = run_dof_diagnosis(
            wrapped_env, dof_idx=0, wave_type=wave_type, period=100, amplitude=1.0, offset=0.0
        )
        plot_metric_on_axis(
            axes[i],
            np.arange(len(target_dof_pos_list)),
            [target_dof_pos_list, dof_pos_list],
            ["Target", "Dof Pos"],
            "Dof Pos",
            wave_type,
            yscale="linear",
        )

    # Save plot
    plot_dir = Path("./gif") / "latest"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / f"action_log.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Action difference plot saved to: {plot_path}")


if __name__ == "__main__":
    fire.Fire(main)
