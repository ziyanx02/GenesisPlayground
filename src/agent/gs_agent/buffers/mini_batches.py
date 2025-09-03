import torch
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel



class OnPolicyMiniBatch(BaseModel):
    """A mini-batch of on-policy transitions for training.

    This provides a unified interface for sampling mini-batches from a rollout buffer and then
    using them for training the policy and value function.
    """

    model_config = genesis_pydantic_config(arbitrary_types_allowed=True)

    obs: torch.Tensor
    act: torch.Tensor
    rew: torch.Tensor
    done: torch.Tensor
    value: torch.Tensor
    log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor



class ImitationMiniBatch(BaseModel):
    """A mini-batch of imitation transitions for training.

    This provides a unified interface for sampling mini-batches from an imitation buffer and then
    using them for training the policy.
    """

    model_config = genesis_pydantic_config(arbitrary_types_allowed=True)

    obs: torch.Tensor
    act: torch.Tensor