import json

# Add examples to path to import utils
import sys
import time
from pathlib import Path
from typing import Any

import fire
import redis
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from gs_env.common.utils.motion_utils import MotionLib
from gs_env.sim.envs.config.schema import MotionEnvArgs
from utils import yaml_to_config  # type: ignore


def _to_list(t: torch.Tensor) -> list[float]:
    return t.detach().cpu().flatten().tolist()


def load_motion_file_from_exp(exp_name: str) -> str:
    deploy_dir = Path(__file__).parent / "logs" / exp_name
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)
    return env_args.motion_file


def publish_motion(
    motion_file: str,
    redis_url: str = "redis://localhost:6379/0",
    key: str = "ref:",
    motion_id: int = 0,
    freq_hz: float = 50.0,
    device: str = "cpu",
) -> None:
    f"""Publish reference motion frames to Redis at a fixed rate.

    The publisher writes each field as a separate Redis key:
      - {key}:motion:base_pos [3]
      - {key}:motion:base_quat [4] (w, x, y, z)
      - {key}:motion:base_lin_vel [3]
      - {key}:motion:base_ang_vel [3]
      - {key}:motion:base_ang_vel_local [3]
      - {key}:motion:dof_pos [D]
      - {key}:motion:dof_vel [D]
      - {key}:motion:link_pos_local [N*3]
      - {key}:motion:link_quat_local [N*4]
      - {key}:motion:foot_contact [F]
      - {key}:timestamp:base_pos [1]
      - {key}:timestamp:base_quat [1]
      - {key}:timestamp:base_lin_vel [1]
      - {key}:timestamp:base_ang_vel [1]
      - {key}:timestamp:base_ang_vel_local [1]
      - {key}:timestamp:dof_pos [1]
      - {key}:timestamp:dof_vel [1]
      - {key}:timestamp:link_pos_local [1]
      - {key}:timestamp:link_quat_local [1]
      - {key}:timestamp:foot_contact [1]
    """
    r = redis.from_url(redis_url)

    device_t = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
    motion_lib = MotionLib(motion_file=motion_file, device=device_t)

    motion_id_t = torch.tensor([motion_id], dtype=torch.long, device=device_t)
    dt = 1.0 / motion_lib.fps

    # Match publishing rate to either provided freq or motion dt, whichever is smaller to avoid skipping
    publish_dt = 1.0 / freq_hz if freq_hz > 0 else dt
    t_val = 0.0

    print("=" * 80)
    print("Motion Publisher started")
    print(f"Redis: {redis_url}")
    print(f"Key: {key}")
    print(f"Motion file: {motion_file}")
    print(f"Motion id: {motion_id}")
    print(f"Publish rate: {1.0 / publish_dt:.2f} Hz")
    print("=" * 80)

    timestamp = 0

    try:
        last_ts = time.time()
        while True:
            # Advance time, loop by motion length
            t_now = time.time()
            if t_now - last_ts < publish_dt:
                time.sleep(0.001)
                continue
            last_ts = t_now
            t_val += publish_dt

            if t_val > motion_lib.get_motion_length(motion_id_t):
                t_val = 0.0
                motion_id_t += 1
                if motion_id_t >= motion_lib.num_motions:
                    motion_id_t = torch.tensor([0], dtype=torch.long, device=device_t)
            motion_time_t = torch.tensor([t_val], dtype=torch.float32, device=device_t)
            (
                base_pos,
                base_quat,
                base_lin_vel,
                base_ang_vel,
                base_ang_vel_local,
                dof_pos,
                dof_vel,
                link_pos_local,
                link_quat_local,
                foot_contact,
            ) = motion_lib.get_ref_motion_frame(
                motion_ids=motion_id_t, motion_times=motion_time_t
            )

            # Publish each field as a separate Redis key
            r.set(f"{key}:motion:base_pos", json.dumps(_to_list(base_pos)))
            r.set(f"{key}:motion:base_quat", json.dumps(_to_list(base_quat)))  # (w, x, y, z)
            r.set(f"{key}:motion:base_lin_vel", json.dumps(_to_list(base_lin_vel)))
            r.set(f"{key}:motion:base_ang_vel", json.dumps(_to_list(base_ang_vel)))
            r.set(f"{key}:motion:base_ang_vel_local", json.dumps(_to_list(base_ang_vel_local)))
            r.set(f"{key}:motion:dof_pos", json.dumps(_to_list(dof_pos)))
            r.set(f"{key}:motion:dof_vel", json.dumps(_to_list(dof_vel)))
            r.set(f"{key}:motion:link_pos_local", json.dumps(_to_list(link_pos_local)))
            r.set(f"{key}:motion:link_quat_local", json.dumps(_to_list(link_quat_local)))
            r.set(f"{key}:motion:foot_contact", json.dumps(_to_list(foot_contact)))
            r.set(f"{key}:timestamp:base_pos", timestamp)
            r.set(f"{key}:timestamp:base_quat", timestamp)
            r.set(f"{key}:timestamp:base_lin_vel", timestamp)
            r.set(f"{key}:timestamp:base_ang_vel", timestamp)
            r.set(f"{key}:timestamp:base_ang_vel_local", timestamp)
            r.set(f"{key}:timestamp:dof_pos", timestamp)
            r.set(f"{key}:timestamp:dof_vel", timestamp)
            r.set(f"{key}:timestamp:link_pos_local", timestamp)
            r.set(f"{key}:timestamp:link_quat_local", timestamp)
            r.set(f"{key}:timestamp:foot_contact", timestamp)
            timestamp += 1
    except KeyboardInterrupt:
        print("\nStopping motion publisher...")


def main(
    exp_name: str | None = None,
    motion_file: str | None = None,
    redis_url: str = "redis://localhost:6379/0",
    key: str = "motion:ref:latest",
    motion_id: int = 0,
    freq_hz: float = 50.0,
    device: str = "cpu",
) -> None:
    if motion_file is None:
        if exp_name is None:
            raise ValueError("Either exp_name or motion_file must be provided")
        motion_file = load_motion_file_from_exp(exp_name)

    publish_motion(
        motion_file=motion_file,
        redis_url=redis_url,
        key=key,
        motion_id=motion_id,
        freq_hz=freq_hz,
        device=device,
    )


if __name__ == "__main__":
    fire.Fire(main)
