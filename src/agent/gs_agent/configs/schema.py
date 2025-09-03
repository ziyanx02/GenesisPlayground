from pydantic.dataclasses import dataclass

from gs_agent.configs.schema import genesis_pydantic_config


@dataclass(config=genesis_pydantic_config(frozen=False))
class AlgoConfig:
    value_coef: float = 1.0  # Value loss coefficient
    ent_coef: float = 0.0  # Entropy coefficient
    clip_param: float = 0.2
    clip_value_loss: bool = True
    gae_gamma: float = 0.98
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    schedule: str = "fixed"
    desired_kl: float = 0.01
    num_epochs: int = 10
    num_mini_batches: int = 4


@dataclass(config=genesis_pydantic_config(frozen=False))
class PolicyConfig:
    # actor-critic (mlp head)
    activation: str = "relu"
    actor_hidden: tuple[int, ...] = (256, 256, 128)
    critic_hidden: tuple[int, ...] = (256, 256, 128)
    init_noise_std: float = 1.0
    norm_obs: bool = False

    # rnn
    use_rnn: bool = False
    rnn_type: str = "gru"  # "lstm" / "gru"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1

    # cnn
    use_cnn: bool = False


@dataclass(config=genesis_pydantic_config(frozen=False))
class RunnerConfig:
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 100
    log_interval: int = 10
    logger: str = "tensorboard"

    #
    video_record: bool = False
    video_interval: int = 100
    RESOLUTION: tuple[int, int] = (320, 240)
    video_folder: str = "videos"


@dataclass(config=genesis_pydantic_config(frozen=False))
class PPOConfig:
    algo: AlgoConfig = AlgoConfig()
    policy: PolicyConfig = PolicyConfig()
    runner: RunnerConfig = RunnerConfig()


def get_ppo_config() -> PPOConfig:
    """
    Returns a default PPO configuration.
    """
    return PPOConfig()
