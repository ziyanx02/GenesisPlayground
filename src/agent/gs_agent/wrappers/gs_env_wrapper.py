import torch
from typing import Any, Mapping, TypeVar, Final
from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")

TGSEnv = TypeVar("TGSEnv")

class GenesisEnvWrapper(BaseEnvWrapper):

    def __init__(
        self,
        env: TGSEnv, device: torch.device = _DEFAULT_DEVICE,
    ) -> None:
        super().__init__(env, device)
        self._curr_obs: torch.Tensor = torch.tensor(self.env.reset()[0], device=self.device)

    # ---------------------------
    # BatchEnvWrapper API (batch)
    # ---------------------------
    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        obs, info = self.env.reset()
        self._curr_obs = torch.tensor(obs, device=self.device)
        return obs, info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        # apply action
        self.env.apply_action(action)
        # get observations
        next_obs = self.env.get_observations()
        # get reward
        reward, reward_terms = self.env.get_reward()
        # get terminated
        terminated = self.env.get_terminated()
        # get truncated
        truncated = self.env.get_truncated()
        # reset if terminated or truncated
        done_idx = terminated | truncated
        if len(done_idx) > 0:
            self.env.reset_idx(done_idx)
        # get extra infos
        extra_infos = self.env.get_extra_infos() 
        extra_infos["reward_terms"] = reward_terms
        return next_obs, reward, terminated, truncated, extra_infos

    def get_observations(self) -> torch.Tensor:
        return self._curr_obs

    @property
    def action_dim(self) -> int:
        return self.env.action_dim
    
    @property
    def actor_obs_dim(self) -> int:
        return self.env.actor_obs_dim
    
    @property
    def critic_obs_dim(self) -> int:
        return self.env.critic_obs_dim

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def close(self) -> None:
        self.env.close()

    def render(self) -> None:
        self.env.render()