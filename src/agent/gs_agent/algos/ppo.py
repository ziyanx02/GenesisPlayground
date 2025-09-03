import os
import statistics
import time
from collections import deque
from typing import Final, Any
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from gs_agent.buffers.gae_buffer import GAEBuffer
from gs_agent.buffers.transition import PPOTransition
from gs_agent.configs.schema import PPOArgs
from gs_agent.modules.policies import GaussianPolicy
from gs_agent.modules.critics import StateValueFunction, QValueFunction
from gs_agent.modules.models import NetworkFactory
from gs_agent.bases.algo import BaseAlgo
from gs_agent.bases.env_wrapper import BaseEnvWrapper

_DEFAULT_DEVICE: Final[torch.device] = torch.device("cpu")
_DEQUE_MAXLEN: Final[int] = 100
"""Max length of the deque for storing episode statistics."""

class PPO(BaseAlgo):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """

    def __init__(self, env: BaseEnvWrapper, cfg: PPOArgs, device: torch.device = _DEFAULT_DEVICE):
        super().__init__(env, cfg, device)
        self._actor_obs_dim = self.env.actor_obs_dim
        self._critic_obs_dim = self.env.critic_obs_dim
        self._action_dim = self.env.action_dim

        self._num_envs = self.env.num_envs
        self._num_steps = cfg.rollout_length

        self.current_iter = 0
        self.desired_kl = self.cfg.target_kl

        self._rewbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._lenbuffer = deque(maxlen=_DEQUE_MAXLEN)
        self._curr_reward_sum = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)
        self._curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)
        #
        self._build_actor_critic()
        self._build_rollouts()

    def _build_actor_critic(self):
        policy_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.policy.policy_backbone,
            input_dim=self._actor_obs_dim,
            output_dim=self._action_dim,
            device=self.device,
        )
        print(f"Policy backbone: {policy_backbone}")

        self._actor = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device) # type: ignore
        self._actor_optimizer = torch.optim.Adam(self._actor.parameters(), lr=self.cfg.lr)

        critic_backbone = NetworkFactory.create_network(
            network_backbone_args=self.cfg.critic.critic_backbone,
            input_dim=self._critic_obs_dim,
            output_dim=1,
            device=self.device,
        )
        print(f"Critic backbone: {critic_backbone}")
        self._critic = StateValueFunction(critic_backbone).to(self.device)
        self._critic_optimizer = torch.optim.Adam(self._critic.parameters(), lr=self.cfg.value_lr)

        self.actor_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization
        self.critic_obs_normalizer = torch.nn.Identity().to(self.device)  # no normalization

    def _build_rollouts(self):
        self._rollouts = GAEBuffer(
            num_envs=self._num_envs,
            max_steps=self._num_steps,
            actor_obs_size=self._actor_obs_dim,
            critic_obs_size=self._critic_obs_dim,
            action_size=self._action_dim,
            device=self.device,
        )

    def _collect_rollouts(self, num_steps: int) -> dict[str, Any]:
        with torch.inference_mode():
            # collect rollouts and compute returns & advantages
            start = time.time()
            for step in range(self._num_steps):
                #
                transition = PPOTransition(
                    actor_obs=self.actor_obs_normalizer(mini_batch["actor_obs"]),
                    critic_obs=self.critic_obs_normalizer(critic_obs),
                )
                # Step environment
                next_obs, reward, done, extra_infos = self.env.step(transition.actions)
                reward_dict[step] = extra_infos["reward_dict"]

                # normalize observations
                critic_obs = extra_infos["observations"].get("critic", obs)
                obs = next_obs

                self._update_transition(reward, done, extra_infos, transition)

                if logger is not None: # type: ignore
                    curr_reward_sum += reward
                    curr_ep_len += 1
                    new_ids = (done > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(curr_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(curr_ep_len[new_ids][:, 0].cpu().numpy().tolist())
                    curr_reward_sum[new_ids] = 0
                    curr_ep_len[new_ids] = 0
            #
            stop = time.time()
            collection_time = stop - start
            #
            start = stop
            last_value = self._critic(critic_obs).detach()
            self._rollouts.compute_gae(last_value)

    def train_one_batch(self, mini_batch: dict[str, Any]) -> dict[str, Any]:
        old_log_probs = torch.squeeze(mini_batch["action_logprobs"])

        new_log_probs = self._actor.evaluate_log_prob(mini_batch["actor_obs"], mini_batch["actions"])
        batch_adv = mini_batch["advantages"]
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = -torch.squeeze(batch_adv) * ratio
        surr2 = -torch.squeeze(batch_adv) * torch.clamp(
            ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio
        )
        policy_loss = torch.max(surr1, surr2).mean()

        # TODO: add KL divergence loss
        # if self.desired_kl is not None and self.schedule == "adaptive":
        #     with torch.inference_mode():
        #         kl = torch.sum(
        #             torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
        #             + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
        #             / (2.0 * torch.square(sigma_batch))
        #             - 0.5,
        #             axis=-1,
        #         )
        #         kl_mean = torch.mean(kl)
        #
        #         if kl_mean > self.desired_kl * 2.0:
        #             self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        #         elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
        #             self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        #
        #         for param_group in self.optimizer.param_groups:
        #             param_group["lr"] = self.learning_rate

        # Calculate value loss
        values = self._critic(mini_batch["critic_obs"])
        if self.cfg.clip_value_loss:
            value_clipped = mini_batch["values"] + (values - mini_batch["values"]).clamp(
                -self.cfg.clip_param, self.cfg.clip_param
            )
            value_losses = (values - mini_batch["returns"]).pow(2)
            value_losses_clipped = (value_clipped - mini_batch["returns"]).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (mini_batch["returns"] - values).pow(2).mean()

        # Calculate entropy loss
        entropy = self._actor.entropy_on(mini_batch["actor_obs"])
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            - self.cfg.entropy_coef * entropy_loss
        )

        # Optimization step
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self._actor.parameters(), self.cfg.max_grad_norm)
        nn.utils.clip_grad_norm_(self._critic.parameters(), self.cfg.max_grad_norm)
        self._critic_optimizer.step()
        self._actor_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def train_one_episode(self) -> dict[str, Any]:
        # collect rollouts
        rollouts = self._collect_rollouts(num_steps=self._num_steps)

        generator = self._rollouts.minibatch_gen(
            num_mini_batches=self.cfg.num_mini_batches,
            num_epochs=self.cfg.num_epochs,
        )
        metrics = {}
        for mini_batch in generator:
            metrics = self.train_one_batch(mini_batch)
            metrics.update(metrics)

        self._rollouts.reset()
        return metrics




    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self._actor.state_dict(),
            "actor_optimizer_state_dict": self._actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self._critic_optimizer.state_dict(),
            "iter": self.current_iter,
        }
        if self.cfg.norm_obs:
            saved_dict["actor_obs_normalizer"] = self.actor_obs_normalizer.state_dict()
            saved_dict["critic_obs_normalizer"] = self.critic_obs_normalizer.state_dict()
        if infos is not None:
            saved_dict.update(infos)
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path)
        self._actor.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
            self._critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        self._current_iter = checkpoint["iter"]
        if self.cfg.norm_obs:
            self.actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer"])
            self.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer"])
        return checkpoint.get("infos", None)

    def train_mode(self):
        self._actor.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    def eval_mode(self):
        self._actor.eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def get_inference_policy(self, device=None):
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        # policy = self._actor_critic.act_inference
        policy = self._actor
        if self.cfg.policy.norm_obs:
            self.actor_obs_normalizer.to(device)
            return policy, self.actor_obs_normalizer
        else:
            return policy, None
