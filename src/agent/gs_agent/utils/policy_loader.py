import glob
import os
import pickle

import torch

from gs_agent.modules.actor_critic import ActorCritic, ActorCriticRecurrent


def load_latest_experiment(exp_name="goal_reach", algo="gsppo"):
    """Load the most recent experiment directory and return its path."""
    log_pattern = f"logs/{algo}_{exp_name}_*"
    log_dirs = glob.glob(log_pattern)

    if not log_dirs:
        raise FileNotFoundError(f"No experiment directories found matching pattern: {log_pattern}")

    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    log_dir = log_dirs[0]

    print(f"Loading from most recent experiment: {log_dir}")
    return log_dir


def load_ppo_config(log_dir):
    """Load training configuration from experiment directory."""
    config_path = os.path.join(log_dir, "ppo_cfg.pkl")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ppo_cfg = pickle.load(open(config_path, "rb"))
    print(f"Loaded training configuration from: {config_path}")
    return ppo_cfg


def load_latest_model(log_dir):
    """Load the most recent model checkpoint from experiment directory."""
    model_files = glob.glob(os.path.join(log_dir, "checkpoints/model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {log_dir}")

    model_files.sort(
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True
    )
    resume_path = model_files[0]

    print(f"Loading model from: {resume_path}")
    return resume_path


def load_policy_from_file(log_dir, device="cpu"):
    """Load policy model from the latest checkpoint in the given log directory."""
    cfg = load_ppo_config(log_dir)
    resume_path = load_latest_model(log_dir)
    checkpoint = torch.load(resume_path, map_location=device)
    if cfg.policy.use_rnn:
        model_cls = ActorCriticRecurrent
        model = model_cls(
            actor_input_dim=14,
            critic_input_dim=14,
            act_dim=6,
            cfg=cfg,
            rnn_type=getattr(cfg.policy, "rnn_type", "lstm"),
            rnn_num_layers=getattr(cfg.policy, "rnn_num_layers", 1),
            rnn_hidden_size=getattr(cfg.policy, "rnn_hidden_size", 256),
        ).to(device)
    else:
        model_cls = ActorCritic
        model = model_cls(
            actor_input_dim=14,
            critic_input_dim=14,
            act_dim=6,
            cfg=cfg,
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
