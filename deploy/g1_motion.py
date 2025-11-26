import platform
import sys
import time
from pathlib import Path

import fire
import torch
from gs_env.sim.envs.config.schema import MotionEnvArgs
from gs_env.common.utils.math_utils import quat_apply, quat_from_angle_axis, quat_mul

# Add examples to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from deploy.utils import RedisClient
from examples.utils import yaml_to_config  # type: ignore


def load_checkpoint_and_env_args(
    exp_name: str, num_ckpt: int | None = None, device: str = "cuda"
) -> tuple[torch.jit.ScriptModule, MotionEnvArgs]:
    """Load JIT checkpoint and env_args from deploy/logs directory.

    Args:
        exp_name: Experiment name
        num_ckpt: Checkpoint number. If None, loads the latest checkpoint.

    Returns:
        Tuple of (checkpoint_path, env_args)
    """

    deploy_dir = Path(__file__).parent / "logs" / exp_name
    if not deploy_dir.exists():
        raise FileNotFoundError(f"Deploy directory not found: {deploy_dir}")

    # Load env_args from YAML
    env_args_path = deploy_dir / "env_args.yaml"
    if not env_args_path.exists():
        raise FileNotFoundError(f"env_args.yaml not found: {env_args_path}")

    print(f"Loading env_args from: {env_args_path}")
    env_args = yaml_to_config(env_args_path, MotionEnvArgs)

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
    # Load policy
    policy = torch.jit.load(str(ckpt_path))
    policy.to(device)
    policy.eval()

    return policy, env_args


def main(
    exp_name: str = "bc_motion",
    num_ckpt: int | None = None,
    device: str = "cpu",
    show_viewer: bool = True,
    sim: bool = True,
    view: bool = False,
    action_scale: float = 0.0,  # only for real robot
    redis_url: str = "redis://localhost:6379/0",
    redis_key: str = "motion:ref:latest",
) -> None:
    """Run BC motion policy on either simulation or real robot.

    This script deploys policies trained by run_bc_motion.py.

    Args:
        exp_name: Experiment name (subdirectory in deploy/logs)
        num_ckpt: Checkpoint number. If None, loads latest.
        device: Device for policy inference ('cuda' or 'cpu')
        show_viewer: Show viewer (only for sim mode)
        sim: If True, run in simulation. If False, run on real robot.
        view: If True, view motion reference data instead of running policy
        action_scale: Action scaling factor (only for real robot)
        redis_url: Redis URL for motion reference data
        redis_key: Redis key for motion reference data
    """
    device = "cpu" if not torch.cuda.is_available() else device

    # Load checkpoint and env_args
    policy, env_args = load_checkpoint_and_env_args(exp_name, num_ckpt, device)

    if sim:
        print("Running in SIMULATION mode")
        import gs_env.sim.envs as envs

        envclass = getattr(envs, env_args.env_name)
        env = envclass(
            args=env_args,
            num_envs=1,
            show_viewer=show_viewer,
            device=torch.device(device),
            eval_mode=True,
        )
        env.eval()
        env.reset()

    else:
        if view:
            raise ValueError("View mode is only supported in simulation mode")
        print("Running in REAL ROBOT mode")
        from gs_env.real import UnitreeLeggedEnv

        env = UnitreeLeggedEnv(
            env_args, action_scale=action_scale, interactive=True, device=torch.device(device)
        )

        print("Press Start button to start the policy")
        while not env.robot.Start:
            time.sleep(0.1)

    # Get link names and tracking link names for Redis client
    link_names: list[str] = []
    if hasattr(env, "robot"):
        link_names = getattr(env.robot, "link_names", [])
    tracking_link_names = getattr(env_args, "tracking_link_names", [])
    redis_client = RedisClient(
        url=redis_url,
        key=redis_key,
        device=device,
        link_names=link_names,
        tracking_link_names=tracking_link_names,
    )

    if view and sim:
        print("=" * 80)
        print("Starting motion visualization")
        print(f"Mode: SIMULATION (VIEW)")
        print(f"Device: {device}")
        print(f"Redis URL: {redis_url}")
        print(f"Redis Key: {redis_key}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Starting policy execution")
        print(f"Mode: {'SIMULATION' if sim else 'REAL ROBOT'}")
        print(f"Device: {device}")
        print("=" * 80)

    def view_loop() -> None:
        """View motion reference data from Redis."""

        nonlocal env, redis_client
        assert sim, "View mode only works in simulation"
        assert hasattr(env, "scene"), "Environment must have scene attribute for view mode"

        link_name_to_idx = {}
        for link_name in env.scene.objects.keys():  # type: ignore
            if link_name in link_names:
                link_name_to_idx[link_name] = link_names.index(link_name)

        last_update_time = time.time()

        while True:
            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()

            # Update reference values from Redis
            redis_client.update()

            # Transform link positions from local_yaw to global coordinates
            env.robot.set_state(
                pos=redis_client.ref_base_pos,
                quat=redis_client.ref_base_quat,
                dof_pos=redis_client.ref_dof_pos,
                dof_vel=redis_client.ref_dof_vel,
                lin_vel=redis_client.ref_base_lin_vel_local,
                ang_vel=redis_client.ref_base_ang_vel_local,
            )
            ref_quat_yaw = quat_from_angle_axis(
                redis_client.ref_base_euler[:, 2],
                torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
            )
            for link_name in env.scene.objects.keys():  # type: ignore
                if link_name in link_name_to_idx:
                    link_idx = link_name_to_idx[link_name]
                    if link_idx < redis_client.ref_link_pos_local_yaw.shape[1]:
                        ref_link_pos = redis_client.ref_link_pos_local_yaw[:, link_idx, :]
                        ref_link_quat = redis_client.ref_link_quat_local_yaw[:, link_idx, :]
                        ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                        ref_link_pos[:, :2] += redis_client.ref_base_pos[:, :2]
                        ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                        env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)  # type: ignore

            env.scene.scene.step(refresh_visualizer=False)  # type: ignore

    def deploy_loop() -> None:
        nonlocal env, redis_client
        # Initialize tracking variables
        last_action_t = torch.zeros(1, env.action_dim, device=device)
        commands_t = torch.zeros(1, 3, device=device)
        last_update_time = time.time()
        total_inference_time = 0
        step_id = 0

        if sim:
            link_name_to_idx = {}
            for link_name in env.scene.objects.keys():  # type: ignore
                if link_name in link_names:
                    link_name_to_idx[link_name] = link_names.index(link_name)

        # Connect to Redis and construct client
        dof_dim_cfg = len(getattr(env_args.robot_args, "dof_names", []))
        if dof_dim_cfg <= 0:
            dof_dim_cfg = int(last_action_t.shape[-1])

        redis_client.update()
        redis_client.update_quat(env.base_quat)

        while True:
            # Check termination condition (only for real robot)
            if not sim and hasattr(env, "is_emergency_stop") and env.is_emergency_stop:  # type: ignore
                print("Emergency stop triggered!")
                break

            # Control loop timing (50 Hz)
            if time.time() - last_update_time < 0.02:
                time.sleep(0.001)
                continue
            last_update_time = time.time()

            if not sim:
                commands_t[0, 0] = env.robot.Ly  # forward velocity (m/s)
                commands_t[0, 1] = -env.robot.Lx  # lateral velocity (m/s)
                commands_t[0, 2] = -env.robot.Rx  # angular velocity (rad/s)
            else:
                # Update commands (can be modified for different behaviors)
                commands_t[0, 0] = 0.0  # forward velocity (m/s)
                commands_t[0, 1] = 0.0  # lateral velocity (m/s)
                commands_t[0, 2] = 0.0  # angular velocity (rad/s)

            # Update reference values from Redis (zeros if unavailable)
            redis_client.update()

            # Construct observation (matching training observation structure)
            obs_components = []
            for key in env_args.actor_obs_terms:
                if key == "last_action":
                    obs_gt = last_action_t
                elif key == "commands":
                    obs_gt = commands_t
                elif key.startswith("ref_"):
                    obs_gt = (getattr(redis_client, key) * env_args.obs_scales.get(key, 1.0)).reshape(1, -1)
                else:
                    obs_gt = getattr(env, key) * env_args.obs_scales.get(key, 1.0)
                obs_components.append(obs_gt)
            obs_t = torch.cat(obs_components, dim=-1)

            # Get action from policy
            assert policy is not None, "Policy must be loaded for deploy mode"
            with torch.no_grad():
                start_time = time.time()
                action_t = policy(obs_t)
                end_time = time.time()
                total_inference_time += end_time - start_time

            env.apply_action(action_t)

            if sim:
                terminated = env.get_terminated()  # type: ignore
                if terminated[0]:
                    env.reset_idx(torch.IntTensor([0]))  # type: ignore
                for link_name in env.scene.objects.keys():  # type: ignore
                    ref_quat_yaw = quat_from_angle_axis(
                        redis_client.ref_base_euler[:, 2],
                        torch.tensor([0, 0, 1], device=env.device, dtype=torch.float),
                    )
                    if link_name in link_name_to_idx:
                        link_idx = link_name_to_idx[link_name]
                        if link_idx < redis_client.ref_link_pos_local_yaw.shape[1]:
                            ref_link_pos = redis_client.ref_link_pos_local_yaw[:, link_idx, :]
                            ref_link_quat = redis_client.ref_link_quat_local_yaw[:, link_idx, :]
                            ref_link_pos = quat_apply(ref_quat_yaw, ref_link_pos)
                            ref_link_pos[:, :2] += redis_client.ref_base_pos[:, :2]
                            ref_link_quat = quat_mul(ref_quat_yaw, ref_link_quat)
                            env.scene.set_obj_pose(link_name, pos=ref_link_pos, quat=ref_link_quat)  # type: ignore


            last_action_t = action_t.clone()
            step_id += 1

            if step_id % 100 == 0:
                print(f"Step {step_id}: Average inference time: {total_inference_time / 100:.4f}s")
                total_inference_time = 0

    try:
        if view and sim:
            # View mode - show motion from Redis
            if platform.system() == "Darwin" and show_viewer:
                import threading

                threading.Thread(target=view_loop).start()
                env.scene.scene.viewer.run()  # type: ignore
            else:
                view_loop()
        else:
            # Deploy mode - run policy
            if platform.system() == "Darwin" and sim and show_viewer:
                import threading

                threading.Thread(target=deploy_loop).start()
                env.scene.scene.viewer.run()  # type: ignore
            else:
                deploy_loop()
    except KeyboardInterrupt:
        if not sim:
            env.emergency_stop()
        print("\nKeyboardInterrupt received, stopping...")
    finally:
        if not sim:
            print("Stopping robot handler...")
            # Handler cleanup if needed
        else:
            print("Simulation stopped.")


if __name__ == "__main__":
    fire.Fire(main)
