import torch


class PPOTransition:
    def __init__(self):
        self.actor_obs: torch.Tensor
        self.actions: torch.Tensor
        self.critic_obs: torch.Tensor
        self.rewards: torch.Tensor
        self.dones: torch.Tensor
        self.values: torch.Tensor
        self.actions_log_prob: torch.Tensor
        self.action_mean: torch.Tensor
        self.action_sigma: torch.Tensor

        #
        self.actor_hidden = None
        self.critic_hidden = None
        self.depth_obs = None
        self.rgb_obs = None

    def clear(self):
        self.__init__()