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
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
