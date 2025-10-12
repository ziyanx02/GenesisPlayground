#!/usr/bin/env python3
"""Example: Train PPO on Genesis Walking environment using Genesis RL."""

import platform
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_agent.wrappers.gs_env_wrapper import GenesisEnvWrapper
from gs_env.sim.envs.config.registry import EnvArgsRegistry
import gs_env.sim.envs as gs_envs
from utils import plot_metric_on_axis


def create_gs_env(
    show_viewer: bool = False,
    num_envs: int = 4096,
    device: str = "cuda",
    args: Any = None,
    eval_mode: bool = False,
) -> gs_envs.WalkingEnv:
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


def run_single_dof_wave_diagnosis(
    env: GenesisEnvWrapper,
    dof_idx: int = 0,
    num_dofs: int = 29,
    wave_type: str = "SIN",
    period: int = 1,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Run single DoF diagnosis."""
    print(f"Running DoF diagnosis for {wave_type}")
    NUM_TOTAL_PERIODS = 5

    def linear_phase(x: int) -> float:
        return x / period * 2 * np.pi

    def quadratic_phase(x: int) -> float:
        return (x / period) ** 2 * 2 * np.pi / NUM_TOTAL_PERIODS * 3

    def linear_amp(x: int) -> float:
        return (10 * (x - 1) / period) // NUM_TOTAL_PERIODS * amplitude / 10

    if wave_type == "SIN":

        def wave_func(x: int) -> float:
            return np.sin(linear_phase(x)) * amplitude
    elif wave_type == "FM-SIN":

        def wave_func(x: int) -> float:
            return np.sin(quadratic_phase(x)) * amplitude
    elif wave_type == "SQ":

        def wave_func(x: int) -> float:
            return np.sign(np.sin(linear_phase(x))) * linear_amp(x)
    else:
        raise ValueError(f"Invalid wave type: {wave_type}")

    env.reset()
    action = torch.zeros((env.num_envs, num_dofs), device=env.device)
    action[:, dof_idx] = offset / env.env.action_scale

    for _ in range(10):
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

        if i % period == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    return target_dof_pos_list, dof_pos_list


def run_single_dof_diagnosis(
    env: GenesisEnvWrapper,
    dof_idx: int = 0,
    dof_name: str = "DoF",
    num_dofs: int = 29,
    period: int = 100,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    for i, wave_type in enumerate(["SIN", "FM-SIN", "SQ"]):
        target_dof_pos_list, dof_pos_list = run_single_dof_wave_diagnosis(
            env,
            dof_idx=dof_idx,
            num_dofs=num_dofs,
            wave_type=wave_type,
            period=period,
            amplitude=amplitude,
            offset=offset,
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

        if wave_type == "SQ":
            truncated_dof_pos_list = dof_pos_list[-period : -period + 10]
            truncated_target_dof_pos_list = target_dof_pos_list[-period : -period + 10]
            plot_metric_on_axis(
                axes[3],
                np.arange(len(truncated_target_dof_pos_list)),
                [truncated_target_dof_pos_list, truncated_dof_pos_list],
                ["Target", "Dof Pos"],
                "Dof Pos",
                "Truncated",
                yscale="linear",
                show_mean=False,
            )

    # Save plot
    plot_dir = Path("./logs") / "pd_test"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_path = plot_dir / f"{dof_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")


def main(
    show_viewer: bool = False,
    device: str = "cuda",
    env_name: str = "g1_fixed",
) -> None:
    # Create environment for evaluation
    env_args = EnvArgsRegistry[env_name]
    env = create_gs_env(
        show_viewer=show_viewer,
        num_envs=1,
        device=device,
        args=env_args,
        eval_mode=True,
    )

    wrapped_env = GenesisEnvWrapper(env, device=env.device)

    # DoF names, lower bound, upper bound
    test_dofs = {
        # "hip_roll": [0.0, 1.0],
        # "hip_pitch": [-0.5, 0.5],
        # "hip_yaw": [-0.5, 0.5],
        # "knee": [0.0, 1.0],
        "ankle_roll": [-0.2, 0.2],
        "ankle_pitch": [-0.5, 0.5],
        # "waist_yaw": [-1.0, 1.0],
        # "waist_roll": [-0.4, 0.4],
        # "waist_pitch": [-0.4, 0.4],
        # "shoulder_roll": [0.0, 1.0],
        # "shoulder_pitch": [-0.5, 0.5],
        # "shoulder_yaw": [0.0, 1.0],
        # "elbow": [0.0, 1.0],
        # "wrist_roll": [-1.0, 1.0],
        # "wrist_pitch": [-1.0, 1.0],
        # "wrist_yaw": [-1.0, 1.0],
    }

    def run_dof_diagnosis_fixed() -> None:
        nonlocal wrapped_env
        dof_names = wrapped_env.env.robot.dof_names

        for dof_name, (lower_bound, upper_bound) in test_dofs.items():
            for i, name in enumerate(dof_names):
                if dof_name in name:
                    dof_idx = i
                    break
            amplitude = (upper_bound - lower_bound) / 2
            offset = (upper_bound + lower_bound) / 2
            run_single_dof_diagnosis(
                wrapped_env,
                dof_idx=dof_idx,
                dof_name=dof_name,
                num_dofs=29,
                period=100,
                amplitude=amplitude,
                offset=offset,
            )

    try:
        if platform.system() == "Darwin" and show_viewer:
            import threading

            threading.Thread(target=run_dof_diagnosis_fixed).start()
            env.scene.scene.viewer.run()  # type: ignore
        else:
            run_dof_diagnosis_fixed()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    fire.Fire(main)
