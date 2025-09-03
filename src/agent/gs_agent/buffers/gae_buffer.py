import torch
from tensordict import TensorDict

from gs_agent.bases.buffer import BaseBuffer
from gs_agent.buffers.transition import OnPolicyTransition
from gs_agent.buffers.mini_batches import OnPolicyMiniBatch
from typing import Final


_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")

ACTOR_OBS = "obs"
ACTIONS = "actions"
REWARDS = "rewards"
DONES = "dones"
VALUES = "values"
ACTION_LOGPROBS = "action_logprobs"
ADVANTAGES = "advantages"
RETURNS = "returns"

class GAEBuffer(BaseBuffer[OnPolicyTransition]):
    """
    A fixed-size buffer for storing rollouts for multi-env on-policy training,
    with GAE/Generalized Advantage Estimation.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        actor_obs_size: int,
        action_size: int,
        device: torch.device = _DEFAULT_DEVICE,
        gae_gamma=0.98,
        gae_lam=0.95,
    ) -> None:
        """
        Args:
            num_envs (int): Number of parallel environments.
            max_steps (int): Maximum number of steps to store in the buffer.
            actor_obs_size (int): Dimension of the actor's observation space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to store the buffer on (default: CPU).
            gae_gamma (float): Discount factor for GAE.
            gae_lam (float): Lambda parameter for GAE.
        """
        super().__init__()
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._actor_obs_dim = actor_obs_size
        self._action_dim = action_size
        self._device = device

        # GAE parameters (configurable)
        self._gae_gamma = gae_gamma
        self._gae_lam = gae_lam

        self._normalize_adv = True  # Whether to normalize advantages

        # Track current size dynamically
        self._idx, self._idx_hidden = 0, 0

        # Initialize buffer
        self._buffer = self._init_buffers(actor_obs_size, action_size)

    def _init_buffers(self, actor_obs_dim: int, action_dim: int) -> TensorDict:
        max_steps, num_envs = self._max_steps, self._num_envs
        buffer = TensorDict(
            {
                ACTOR_OBS: torch.zeros(max_steps, num_envs, actor_obs_dim),
                ACTIONS: torch.zeros(max_steps, num_envs, action_dim),
                REWARDS: torch.zeros(max_steps, num_envs, 1),
                DONES: torch.zeros(max_steps, num_envs, 1).byte(),
                VALUES: torch.zeros(max_steps, num_envs, 1),
                ACTION_LOGPROBS: torch.zeros(max_steps, num_envs, 1),
                ADVANTAGES: torch.zeros(max_steps, num_envs, 1),
                RETURNS: torch.zeros(max_steps, num_envs, 1),
            },
            batch_size=[self._max_steps, self._num_envs],
            device=self._device,
        )
        return buffer


    def reset(self):
        self._idx = 0

    def append(self, transition: OnPolicyTransition) -> None:
        if self._idx >= self._max_steps:
            raise ValueError(f"Buffer full! Cannot append more than {self._max_steps} steps.")
        idx = self._idx
        self._buffer[ACTOR_OBS][idx] = transition.obs   
        self._buffer[ACTIONS][idx] = transition.act
        self._buffer[REWARDS][idx] = transition.rew
        self._buffer[DONES][idx] = transition.done
        self._buffer[VALUES][idx] = transition.value
        self._buffer[ACTION_LOGPROBS][idx] = transition.log_prob.unsqueeze(-1)

        self._buffer[ADVANTAGES][idx].zero_()  # Initialize advantages to zero
        self._buffer[RETURNS][idx].zero_()  # Initialize returns to zero

        # Increment index
        self._idx += 1

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor) -> None:
        assert self._idx == self._max_steps, "Buffer must be full to compute GAE."
        # shape: (max_steps+1, num_envs, 1)
        vals = torch.cat([self._buffer[VALUES], last_value.unsqueeze(0)], dim=0)
        masks = 1.0 - self._buffer[DONES].float()
        deltas = self._buffer[REWARDS] + self._gae_gamma * masks * vals[1:] - vals[:-1]
        # backward recursion
        advantage = torch.zeros_like(last_value)
        for t in reversed(range(self._max_steps)):
            advantage = deltas[t] + self._gae_gamma * self._gae_lam * masks[t] * advantage
            self._buffer[ADVANTAGES][t] = advantage
        self._buffer[RETURNS] = self._buffer[ADVANTAGES] + self._buffer[VALUES]

        # normalize
        if self._normalize_adv:
            adv_mean = self._buffer[ADVANTAGES].mean()
            adv_std = self._buffer[ADVANTAGES].std()
            self._buffer[ADVANTAGES] = (self._buffer[ADVANTAGES] - adv_mean) / (adv_std + 1e-8)

    def is_full(self):
        return self._idx >= self._max_steps

    def __len__(self):
        return self._idx * self._num_envs

    def minibatch_gen(self, num_mini_batches: int, num_epochs: int):
        buffer_size = self._idx * self._num_envs
        indices = torch.randperm(buffer_size)
        # Calculate the size of each mini-batch
        batch_size = self._idx * self._num_envs // num_mini_batches
        # Shuffle indices for mini-batch sampling
        for _ in range(num_epochs):
            for start in range(0, len(indices), batch_size):
                end = start + batch_size
                mb_indices = indices[start:end]
                # Yield a mini-batch of data
                batch = {
                    key: value.view(-1, *value.shape[2:])[mb_indices]
                    for key, value in self._buffer.items()
                }
                yield OnPolicyMiniBatch(**batch)
