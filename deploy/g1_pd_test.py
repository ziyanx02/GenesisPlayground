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
from gs_env.real import UnitreeLeggedEnv
from gs_env.real.unitree.utils.hf_logger import AsyncHFLogger
from gs_env.sim.envs.config.registry import EnvArgsRegistry
from gs_env.sim.envs.config.schema import LeggedRobotEnvArgs

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from utils import (  # type: ignore
    align_hf_arrays_pos_from_logger,
    align_hf_arrays_vel_from_logger,
    analyze_latency_and_noise,
    plot_metric_on_axis,
    yaml_to_config,
)


def load_env_args(exp_name: str) -> LeggedRobotEnvArgs:
    """Load JIT checkpoint and env_args from deploy/logs directory.

    Args:
        exp_name: Experiment name

    Returns:
        env_args
    """

    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, LeggedRobotEnvArgs)

    return env_args


def run_single_dof_wave_diagnosis(
    env: Any,
    dof_idx: int = 0,
    num_dofs: int = 29,
    wave_type: str = "SIN",
    period: int = 1,
    amplitude: float = 1.0,
    offset: float = 0.0,
    hf_logger: AsyncHFLogger | None = None,
) -> list[float]:
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
    action[:, dof_idx] = offset / env.action_scale

    if isinstance(env, UnitreeLeggedEnv):

        def get_state() -> tuple[int, list[float], list[float], list[float]]:
            nj = env.robot.num_full_dof
            q = list(env.robot.joint_pos[:nj])
            dq = list(env.robot.joint_vel[:nj])
            tau = list(env.robot.torque[:nj])
            return nj, q, dq, tau
    else:

        def get_state() -> tuple[int, list[float], list[float], list[float]]:
            nj = env.robot.action_dim
            q = env.robot.dof_pos[0][:nj].detach().cpu().numpy()
            dq = env.robot.dof_vel[0][:nj].detach().cpu().numpy()
            tau = env.robot.torque[0][:nj].detach().cpu().numpy()
            return nj, q, dq, tau

    last_controller_time = time.time()
    current_dof_pos = env.dof_pos[0] - env.default_dof_pos
    TOTAL_RESET_STEPS = 50

    ctrl_dt = 0.02
    hf_rate = 1000.0
    hf_dt = 1.0 / hf_rate

    for i in range(TOTAL_RESET_STEPS):
        while time.time() - last_controller_time < ctrl_dt:
            time.sleep(0.001)
        last_controller_time = time.time()
        env.apply_action(
            current_dof_pos / env.action_scale * (1 - i / TOTAL_RESET_STEPS)
            + action * (i / TOTAL_RESET_STEPS)
        )

    target_dof_pos_list = []
    dof_pos_list = []

    t0 = time.perf_counter()
    next_ctrl = t0 + ctrl_dt
    next_hf = t0 + hf_dt

    for i in range(period * NUM_TOTAL_PERIODS):
        while True:
            now = time.perf_counter()

            # Flush HF ticks
            while now >= next_hf:
                t_ns = time.perf_counter_ns()
                nj, q, dq, tau = get_state()

                if hf_logger:
                    hf_logger.push(t_ns, q, dq, tau)

                next_hf += hf_dt

            # Control tick
            if now >= next_ctrl:
                break
            time.sleep(max(0.0, min(next_hf, next_ctrl) - now))

        # last_controller_time = time.perf_counter()
        target_dof_pos = wave_func(i) + offset
        action[:, dof_idx] = target_dof_pos / env.action_scale
        dof_pos = env.dof_pos[0, dof_idx].cpu().item() - env.robot.default_dof_pos[dof_idx]
        target_dof_pos_list.append(target_dof_pos)
        dof_pos_list.append(dof_pos)
        env.apply_action(action)

        next_ctrl += ctrl_dt

        if i % period == 0:
            print(f"Step {i} of {period * NUM_TOTAL_PERIODS}")

    return target_dof_pos_list, dof_pos_list


def run_single_dof_diagnosis(
    env: Any,
    dof_idx: int = 0,
    dof_name: str = "DoF",
    num_dofs: int = 29,
    period: int = 100,
    amplitude: float = 1.0,
    offset: float = 0.0,
    hf_logging: bool = False,
    log_dir: Path = Path(__file__).parent / "logs" / "pd_test",
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(12, 12))
    hf_logger = None
    for i, wave_type in enumerate(["SIN", "FM-SIN"]):
        if hf_logging:
            hf_logger = AsyncHFLogger(nj=env.action_dim, max_seconds=10.0, rate_hz=1000)
            hf_logger.start()

        target_dof_pos_list, dof_pos_list = run_single_dof_wave_diagnosis(
            env,
            dof_idx=dof_idx,
            num_dofs=num_dofs,
            wave_type=wave_type,
            period=period,
            amplitude=amplitude,
            offset=offset,
            hf_logger=hf_logger,
        )

        if hf_logging:
            hf_logger.stop()
            t_ns, q, _, _ = hf_logger.get()
            q_arr = q - np.asarray(env.robot.default_dof_pos.cpu())
            print(t_ns.shape)
            steps, target_pos_out, q_out = align_hf_arrays_pos_from_logger(
                t_ns=t_ns,
                q_arr=q_arr,
                dof_idx=dof_idx,
                target_pos=target_dof_pos_list,
                control_dt=0.02,  # estimated or configured
                out_rate_hz=200.0,
                pos_offset=0.0,  # centers both streams
            )

            sample_dt = 1.0 / 200.0
            analyze_latency_and_noise(
                target_pos_out,
                q_out,
                sample_dt=sample_dt,
                name=f"{dof_name}-{wave_type}-pos",
                trend_window_s=0.1,  # adjust if you want
            )

            plot_metric_on_axis(
                axes[2 * i],
                steps,
                [target_pos_out.tolist(), q_out.tolist()],
                ["Target", "Dof Pos"],
                "Dof Pos",
                wave_type,
                yscale="linear",
            )

            t_ns, _, dq_arr, _ = hf_logger.get()
            steps, target_vel_out, dq_interp = align_hf_arrays_vel_from_logger(
                t_ns=t_ns,
                dq_arr=dq_arr,
                dof_idx=dof_idx,
                target_pos=target_dof_pos_list,  # the list you already build each control tick
                control_dt=0.02,
                out_rate_hz=200.0,
            )
            analyze_latency_and_noise(
                target_vel_out,
                dq_interp,
                sample_dt=sample_dt,
                name=f"{dof_name}-{wave_type}-vel",
                trend_window_s=0.1,
            )
            plot_metric_on_axis(
                axes[2 * i + 1],
                steps,
                [target_vel_out.tolist(), dq_interp.tolist()],
                ["Target Vel (HF)", "Measured Vel (HF)"],
                ylabel="Joint Velocity (rad/s)",
                title=f"{wave_type} (aligned HF velocity)",
                yscale="linear",
            )
        else:
            plot_metric_on_axis(
                axes[i],
                np.arange(len(target_dof_pos_list)),
                [target_dof_pos_list, dof_pos_list],
                ["Target", "Dof Pos"],
                "Dof Pos",
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

    # Save plot
    log_dir.mkdir(parents=True, exist_ok=True)

    plot_path = log_dir / f"{dof_name}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")


def main(
    show_viewer: bool = False,
    device: str = "cpu",
    sim: bool = True,
    hf_logging: bool = False,
) -> None:
    # Load checkpoint and env_args
    env_args = EnvArgsRegistry["g1_fixed"]

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
            log_dir = Path(__file__).parent / "logs" / "pd_test" / "sim" / f"{env.robot.ctrl_type}"
        else:
            log_dir = Path(__file__).parent / "logs" / "pd_test" / "real" / f"{env.ctrl_type}"

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
                hf_logging=hf_logging,
                log_dir=log_dir,
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
