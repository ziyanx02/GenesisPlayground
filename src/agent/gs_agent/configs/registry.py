from pathlib import Path

from gs_agent.configs.schema import (
    ActivationType,
    MLPConfig,
    OptimizerType,
    PPOArgs,
    RunnerArgs,
)

# ------------------------------------------------------------
# Network Config
# ------------------------------------------------------------

DEFAULT_MLP = MLPConfig(
    hidden_dims=(256, 256, 128),
    activation=ActivationType.RELU,
)


# ------------------------------------------------------------
# Algorithm Config
# ------------------------------------------------------------

# default PPO config
PPO_DEFAULT = PPOArgs(
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=1000,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)

# gym pendulum PPO config
PPO_PENDULUM_MLP = PPOArgs(
    # Network architecture
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    value_lr=None,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=1000,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)


# goal reaching PPO config
PPO_GOAL_REACHING_MLP = PPOArgs(
    policy_backbone=DEFAULT_MLP,
    critic_backbone=DEFAULT_MLP,
    lr=3e-4,
    value_lr=None,
    gamma=0.99,
    gae_lambda=0.95,
    clip_ratio=0.2,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=1.0,
    target_kl=0.02,
    num_epochs=10,
    num_mini_batches=4,
    rollout_length=32,
    optimizer_type=OptimizerType.ADAM,
    weight_decay=0.0,
)


# ------------------------------------------------------------
# Runner Config
# ------------------------------------------------------------


RUNNER_DEFAULT = RunnerArgs(
    total_episodes=100,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/default"),
)


RUNNER_PENDULUM_MLP = RunnerArgs(
    total_episodes=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gym_pendulum"),
)

RUNNER_GOAL_REACHING_MLP = RunnerArgs(
    total_episodes=500,
    log_interval=10,
    save_interval=100,
    save_path=Path("./logs/ppo_gs_goal_reaching"),
)
