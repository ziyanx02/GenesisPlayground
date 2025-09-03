import torch
from gs_schemas.base_types import genesis_pydantic_config
from pydantic import BaseModel
class OnPolicyTransition(BaseModel):
    """A complete transition for on-policy training.

    This transition is used for on-policy training, where the policy is updated based on the
    observed transitions and auxiliary tensors such as value, log_prob.
    """

    model_config = genesis_pydantic_config(arbitrary_types_allowed=True)

    obs: torch.Tensor # [batch, obs_dim]
    act: torch.Tensor # [batch, act_dim]
    rew: torch.Tensor # [batch, 1]
    done: torch.Tensor # [batch, 1]

    # auxiliary tensors
    value: torch.Tensor # [batch, 1]
    log_prob: torch.Tensor # [batch, 1]

    # optional tensors
    critic_obs: torch.Tensor | None = None # [batch, critic_obs_dim]
    rgb_obs: torch.Tensor | None = None # [batch, 3, h, w]
    depth_obs: torch.Tensor | None = None # [batch, 1, h, w]



class ImitationTransition(BaseModel):
    """A complete transition for imitation learning.

    This transition is used for imitation learning, where the policy is updated based on the
    expert demonstrations.
    """

    model_config = genesis_pydantic_config(arbitrary_types_allowed=True)

    obs: torch.Tensor # [batch, obs_dim]
    act: torch.Tensor # [batch, act_dim]        


Transition = OnPolicyTransition | ImitationTransition