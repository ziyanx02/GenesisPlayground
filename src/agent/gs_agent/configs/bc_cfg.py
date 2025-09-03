from pydantic.dataclasses import dataclass

from gs_agent.configs.schema import genesis_pydantic_config


@dataclass(config=genesis_pydantic_config(frozen=False))
class AlgoConfig:
    learning_rate: float = 3e-4
    num_epochs: int = 10
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0


@dataclass(config=genesis_pydantic_config(frozen=False))
class PolicyConfig:
    # actor-critic (mlp head)
    activation: str = "relu"
    actor_hidden: tuple[int, ...] = (256, 256, 128)
    init_noise_std: float = 1.0
    norm_obs: bool = False

    # rnn
    use_rnn: bool = True
    rnn_type: str = "gru"  # "lstm" / "gru"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1

    # cnn
    use_cnn: bool = True
    cnn_output_dim: int = 128


@dataclass(config=genesis_pydantic_config(frozen=False))
class RunnerConfig:
    num_steps_per_env: int = 24
    max_iterations: int = 1000
    save_interval: int = 100
    log_interval: int = 10
    logger: str = "tensorboard"


@dataclass(config=genesis_pydantic_config(frozen=False))
class BCConfig:
    algo: AlgoConfig = AlgoConfig()
    policy: PolicyConfig = PolicyConfig()
    runner: RunnerConfig = RunnerConfig()


def get_bc_config() -> BCConfig:
    """
    Returns a default PPO configuration.
    """
    return BCConfig()
