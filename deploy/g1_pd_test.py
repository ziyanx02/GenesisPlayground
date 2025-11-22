import platform
import sys
import time
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent windows from showing
import matplotlib.pyplot as plt
import numpy as np
import torch
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import (
    plot_metric_on_axis,
    yaml_to_config,
    cross_correlation,
    compute_SRD,
    compute_SD,
)  # type: ignore


def run_single_dof_wave_diagnosis(
    env: Any,
    dof_idx: int = 0,
    num_dofs: int = 29,
    wave_type: str = "SIN",
    period: int = 1,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> dict[str, Any]:
    """Run single DoF diagnosis."""
    print(f"Running DoF diagnosis for {wave_type}")
    NUM_TOTAL_PERIODS = 3

    def linear_phase(x: int) -> float:
        return x / period * 2 * np.pi

    def quadratic_phase(x: int) -> float:
        return (x / period) ** 2 * 2 * np.pi / NUM_TOTAL_PERIODS * 2

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
    action = torch.zeros((1, num_dofs), device=env.device)
    action[:, dof_idx] = offset / env.action_scale[dof_idx]

    last_update_time = time.time()

    current_dof_pos = env.dof_pos[0] - env.default_dof_pos
    TOTAL_RESET_STEPS = 50
    for i in range(TOTAL_RESET_STEPS):
        while time.time() - last_update_time < 0.02:
            time.sleep(0.001)
        last_update_time = time.time()
        env.apply_action(
            current_dof_pos / env.action_scale * (1 - i / TOTAL_RESET_STEPS)
            + action * (i / TOTAL_RESET_STEPS)
        )

    env.robot.start_logging()

    for i in range(period * NUM_TOTAL_PERIODS):
        while time.time() - last_update_time < 0.02:
            time.sleep(0.001)
        last_update_time = time.time()

        target_dof_pos = wave_func(i) + offset
        action[:, dof_idx] = target_dof_pos / env.action_scale[dof_idx]
        env.apply_action(action)

        if i % period == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    log = env.robot.stop_logging()
    for key in log.keys():
        log[key] = log[key][..., dof_idx]
        if type(log[key]) == torch.Tensor:
            log[key] = log[key].squeeze().cpu().numpy()

    return log


def run_single_dof_diagnosis(
    env: Any,
    dof_idx: int = 0,
    dof_name: str = "DoF",
    num_dofs: int = 29,
    period: int = 100,
    amplitude: float = 1.0,
    offset: float = 0.0,
    log_dir: Path = Path(__file__).parent / "logs" / "pd_test",
    sim: bool = True,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    wave_types = ["SIN", "FM-SIN"] if sim else ["SIN"]
    for i, wave_type in enumerate(wave_types):
        log = run_single_dof_wave_diagnosis(
            env,
            dof_idx=dof_idx,
            num_dofs=num_dofs,
            wave_type=wave_type,
            period=period,
            amplitude=amplitude,
            offset=offset,
        )
        target_dof_pos_list = log["target_dof_pos"].tolist()
        dof_pos_list = log["dof_pos"].tolist()
        dof_vel_list = log["dof_vel"].tolist()

        dof_pos_lag = cross_correlation(
            np.array(target_dof_pos_list[::4]), np.array(dof_pos_list[::4])
        ) * env.dt
        print("=" * 40)
        print(f"Lag: {dof_pos_lag:.4f}")
        print("=" * 40)
        print(f"target_dof_pos SRD: {compute_SRD(np.array(target_dof_pos_list)):.4f}, SD: {compute_SD(np.array(target_dof_pos_list)):.4f}")
        print(f"target_dof_pos / 4 SRD: {compute_SRD(np.array(target_dof_pos_list[::4])):.4f}, SD: {compute_SD(np.array(target_dof_pos_list[::4])):.4f}")
        print(f"dof_pos SRD: {compute_SRD(np.array(dof_pos_list)):.4f}, SD: {compute_SD(np.array(dof_pos_list)):.4f}")
        print(f"dof_pos / 4 SRD: {compute_SRD(np.array(dof_pos_list[::4])):.4f}, SD: {compute_SD(np.array(dof_pos_list[::4])):.4f}")
        if sim:
            data_lists = [target_dof_pos_list, dof_pos_list]
            labels = ["Target", "Dof Pos"]
        else:
            dof_pos_raw_list = log["dof_pos_raw"].tolist()
            print("=" * 40)
            dof_pos_lag = cross_correlation(
                np.array(target_dof_pos_list[::4]), np.array(dof_pos_raw_list[::4])
            ) * env.dt
            print(f"Raw Lag: {dof_pos_lag:.4f}")
            print("=" * 40)
            print(f"dof_pos_raw SRD: {compute_SRD(np.array(dof_pos_raw_list)):.4f}, SD: {compute_SD(np.array(dof_pos_raw_list)):.4f}")
            print(f"dof_pos_raw / 4 SRD: {compute_SRD(np.array(dof_pos_raw_list[::4])):.4f}, SD: {compute_SD(np.array(dof_pos_raw_list[::4])):.4f}")
            data_lists = [target_dof_pos_list, dof_pos_list, dof_pos_raw_list]
            labels = ["Target", "Dof Pos", "Dof Pos Raw"]

        plot_metric_on_axis(
            axes[i * 2],
            np.arange(len(target_dof_pos_list)),
            data_lists,
            labels,
            "Dof Pos",
            wave_type,
            yscale="linear",
        )

        print("=" * 40)
        print(f"dof_vel SRD: {compute_SRD(np.array(dof_vel_list)):.4f}, SD: {compute_SD(np.array(dof_vel_list)):.4f}")
        print(f"dof_vel / 4 SRD: {compute_SRD(np.array(dof_vel_list[::4])):.4f}, SD: {compute_SD(np.array(dof_vel_list[::4])):.4f}")

        if sim:
            data_lists = [dof_vel_list]
            labels = ["Dof Vel"]
        else:
            dof_vel_raw_list = log["dof_vel_raw"].tolist()
            print(f"dof_vel_raw SRD: {compute_SRD(np.array(dof_vel_raw_list)):.4f}, SD: {compute_SD(np.array(dof_vel_raw_list)):.4f}")
            data_lists = [dof_vel_list, dof_vel_raw_list]
            labels = ["Dof Vel", "Dof Vel Raw"]

        plot_metric_on_axis(
            axes[i * 2 + 1],
            np.arange(len(target_dof_pos_list)),
            data_lists,
            labels,
            "Dof Vel",
            wave_type,
            yscale="linear",
        )

        # if wave_type == "SQ":
        #     truncated_dof_pos_list = dof_pos_list[-period : -period + 10]
        #     truncated_target_dof_pos_list = target_dof_pos_list[-period : -period + 10]
        #     plot_metric_on_axis(
        #         axes[3],
        #         np.arange(len(truncated_target_dof_pos_list)),
        #         [truncated_target_dof_pos_list, truncated_dof_pos_list],
        #         ["Target", "Dof Pos"],
        #         "Dof Pos",
        #         "Truncated",
        #         yscale="linear",
        #         show_mean=False,
        #     )

        print("=" * 40)

    # Save plot
    log_dir.mkdir(parents=True, exist_ok=True)

    sim_suffix = "sim" if sim else "real"
    plot_path = log_dir / f"{dof_name}_{sim_suffix}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")


def main(
    show_viewer: bool = False,
    device: str = "cpu",
    sim: bool = True,
) -> None:
    # Load checkpoint and env_args
    env_args = EnvArgsRegistry["g1_fixed"]

    robot_args = env_args.robot_args.model_copy(
        update={
            "decimation": 4,
            "low_pass_alpha": 0.3,
            "feed_forward_ratio": 0.5,
        }
    )
    env_args = env_args.model_copy(
        update={
            "robot_args": robot_args,
            "action_latency": 0,
        }
    )

    if sim:
        print("Running in SIMULATION mode")
        from gs_env.sim.envs.locomotion.leggedrobot_env import LeggedRobotEnv

        env = LeggedRobotEnv(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        env.eval()
        env.reset()

    else:
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args, action_scale=1.0, interactive=True, device=torch.device(device)
        )

        print("Press Start button to start the test")
        while not env.robot.Start:
            time.sleep(0.1)

    # DoF names, lower bound, upper bound
    test_dofs = {
        # "hip_roll": [0.0, 0.8],
        # "hip_pitch": [-0.5, 0.5],
        # "hip_yaw": [-0.5, 0.5],
        "knee": [0.0, 1.0],
        # "ankle_roll": [-0.2, 0.2],
        # "ankle_pitch": [-0.5, 0.5],
        # "waist_yaw": [-1.0, 1.0],
        # "waist_roll": [-0.3, 0.3],
        # "waist_pitch": [-0.3, 0.3],
        # "shoulder_roll": [0.0, 1.0],
        # "shoulder_pitch": [-0.5, 0.5],
        # "shoulder_yaw": [0.0, 1.0],
        # "elbow": [0.0, 1.0],
        # "wrist_roll": [-1.0, 1.0],
        # "wrist_pitch": [-1.0, 1.0],
        # "wrist_yaw": [0.0, 1.0],
    }

    def run_dof_diagnosis_fixed() -> None:
        nonlocal env
        dof_names = env.dof_names

        if sim:
            log_dir = Path(__file__).parent / "logs" / "pd_test" / "sim"
        else:
            log_dir = Path(__file__).parent / "logs" / "pd_test" / "real"

        for dof_name, (lower_bound, upper_bound) in test_dofs.items():
            dof_idx = -1
            for i, name in enumerate(dof_names):
                if dof_name in name:
                    dof_idx = i
                    break
            if dof_idx == -1:
                print(f"Dof {dof_name} not found")
                continue
            amplitude = (upper_bound - lower_bound) / 2
            offset = (upper_bound + lower_bound) / 2
            run_single_dof_diagnosis(
                env,
                dof_idx=dof_idx,
                dof_name=dof_name,
                num_dofs=env.action_dim,
                period=100,
                amplitude=amplitude,
                offset=offset,
                log_dir=log_dir,
                sim=sim,
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
