"""Core configuration schemas for the Genesis RL library.

This module defines the fundamental Pydantic dataclasses for configuring
RL algorithms, environments, networks, and training parameters.
"""

from pathlib import Path
from typing import Literal

from gs_schemas.base_types import GenesisEnum, genesis_pydantic_config
from pydantic import BaseModel, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat

# ============================================================================
# Enums
# ============================================================================


class BackendType(GenesisEnum):
    """Backend types for RL computations."""

    CPU = "CPU"
    CUDA = "CUDA"
    MPS = "MPS"


class AlgorithmType(GenesisEnum):
    """Supported RL algorithm types."""

    PPO = "PPO"


class NetworkBackboneType(GenesisEnum):
    MLP = "MLP"
    RNN = "RNN"
    CNN = "CNN"


class OptimizerType(GenesisEnum):
    ADAM = "ADAM"
    ADAMW = "ADAMW"
    SGD = "SGD"


class ActivationType(GenesisEnum):
    RELU = "RELU"
    TANH = "TANH"
    GELU = "GELU"
    SWISH = "SWISH"


class PolicyType(GenesisEnum):
    GAUSSIAN = "GAUSSIAN"


class ValueFunctionType(GenesisEnum):
    STATE_VALUE = "STATE_VALUE"
    Q_VALUE = "Q_VALUE"


# ============================================================================
# Base Configuration
# ============================================================================


class GenesisRLInitArgs(BaseModel):
    """Initialization arguments for Genesis RL library."""

    model_config = genesis_pydantic_config(frozen=True)

    seed: int = 0
    precision: Literal["float32", "float64"] = "float32"
    backend: BackendType | None = None
    """None means auto-detect"""


# ============================================================================
# Network Configuration
# ============================================================================


class MLPConfig(BaseModel):
    """Configuration for Multi-Layer Perceptron networks."""

    model_config = genesis_pydantic_config(frozen=True)

    hidden_dims: tuple[int, ...] = (256, 256, 128)
    activation: ActivationType = ActivationType.RELU

    network_type: NetworkBackboneType = NetworkBackboneType.MLP


NetworkBackboneConfig = MLPConfig


# ============================================================================
# Algorithm Configuration
# ============================================================================


class PPOArgs(BaseModel):
    """Configuration for PPO algorithm."""

    model_config = genesis_pydantic_config(frozen=True)

    # Algorithm type
    algorithm_type: AlgorithmType = AlgorithmType.PPO

    # Network architecture
    policy_backbone: NetworkBackboneConfig = MLPConfig()
    critic_backbone: NetworkBackboneConfig = MLPConfig()

    # Learning rates
    lr: PositiveFloat = 3e-4
    """Policy learning rate"""

    # Value function learning rate
    value_lr: PositiveFloat | None = None
    """None means use the same learning rate as the policy"""

    # Discount and GAE
    gamma: PositiveFloat = Field(default=0.99, ge=0, le=1)
    gae_lambda: PositiveFloat = Field(default=0.95, ge=0, le=1)

    # PPO specific
    clip_ratio: PositiveFloat = 0.2
    value_loss_coef: PositiveFloat = 1.0
    entropy_coef: NonNegativeFloat = 0.0
    max_grad_norm: PositiveFloat = 1.0
    target_kl: PositiveFloat = 0.02

    # Training
    num_epochs: NonNegativeInt = 10
    num_mini_batches: NonNegativeInt = 4
    rollout_length: NonNegativeInt = 1000

    # Optimizer
    optimizer_type: OptimizerType = OptimizerType.ADAM
    weight_decay: NonNegativeFloat = 0.0


AlgorithmArgs = PPOArgs


# ============================================================================
# Runner Configuration
# ============================================================================


class RunnerArgs(BaseModel):
    """Configuration for on-policy runners."""

    model_config = genesis_pydantic_config(frozen=True)

    total_episodes: NonNegativeInt = 1000

    # Training intervals
    log_interval: NonNegativeInt = 10
    save_interval: NonNegativeInt = 100

    save_path: Path = Path(".")
