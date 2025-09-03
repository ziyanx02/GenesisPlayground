import os
import statistics
import time
from collections import deque
from typing import Final
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
from gs_agent.utils.logger import configure as logger_configure
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

        self._actor_critic = GaussianPolicy(
            policy_backbone=policy_backbone,
            action_dim=self._action_dim,
        ).to(self.device) # type: ignore
        self._actor_optimizer = torch.optim.Adam(self._actor_critic.parameters(), lr=self.cfg.lr)

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

    def train(self, num_iters: int, log_dir: str | None = None):
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._actor_optimizer, T_max=num_iters, eta_min=self.cfg.lr * 0.1
        )

        # initialize writer
        logger = logger_configure(folder=log_dir, format_strings=["stdout", "csv", "wandb"])

        # Get initial observations
        obs, extra_infos = self.env.get_observations()
        critic_obs = extra_infos["observations"].get("critic", obs)

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        curr_reward_sum = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)
        curr_ep_len = torch.zeros(self.env.num_envs, device=self.device, dtype=torch.float)

        # Main training loop
        start_iter = self.current_iter
        start_train_time = time.time()
        for it in range(start_iter, self.current_iter + num_iters):
            reward_dict = TensorDict(
                {},
                device=self.device,
                batch_size=[self._num_steps, self._num_envs],
            )
            self.train_mode()  # switch to train mode (for dropout for example)
            with torch.inference_mode():
                # collect rollouts and compute returns & advantages
                start = time.time()
                for step in range(self._num_steps):
                    depth_obs = None
                    if self.cfg.policy.use_cnn:
                        depth_obs = self.env.get_depth_image(normalize=True)
                    #
                    transition = self._get_transition(
                        actor_obs=self.actor_obs_normalizer(obs),
                        critic_obs=self.critic_obs_normalizer(critic_obs),
                        depth_obs=depth_obs,
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

            # Update policy by optimizing PPO objective
            fps = self._num_steps * self._num_envs / collection_time
            mean_pg_loss, mean_value_loss, _ = self.update()
            stop = time.time()
            learn_time = stop - start
            self._current_iter = it

            # Log training statistics
            logger.record("Summary/iters", it)
            if len(rewbuffer) > 0:
                logger.record("Summary/reward_mean", statistics.mean(rewbuffer))
                logger.record("Summary/length_mean", statistics.mean(lenbuffer))
            #
            logger.record("speed/fps", fps)
            logger.record("speed/forward_time", collection_time)
            logger.record("speed/backward_time", learn_time)
            #
            logger.record("train/lr", self._lr_scheduler.get_last_lr()[0])
            logger.record("train/policy_loss", mean_pg_loss)
            logger.record("train/value_loss", mean_value_loss)
            for reward_key in reward_dict.keys():
                reward = torch.mean(reward_dict[reward_key]).item()
                logger.record("train/" + reward_key, reward)

            if it % self._cfg.runner.log_interval == 0:
                logger.dump(step=it)

            if it % self._cfg.runner.save_interval == 0 or it == num_iters - 1:
                # save model
                print(f"Saving model at iteration {it} to {logger.get_dir()}")
                checkpoint_dir = os.path.join(logger.get_dir(), "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                self.save(os.path.join(checkpoint_dir, f"model_{self._current_iter}.pt"))

            #
            self._lr_scheduler.step()
        end_train_time = time.time()
        print(f"Training completed in {end_train_time - start_train_time:.2f} seconds.")

    def _get_transition(self, actor_obs, critic_obs, depth_obs=None, rgb_obs=None) -> PPOTransition:
        transition = PPOTransition()
        #
        if self._actor_critic.is_recurrent:
            # Initialize hidden states for recurrent actor-critic
            transition.actor_hidden, transition.critic_hidden = (
                self._actor_critic.get_hidden_states()
            )

        # Actor features (image or obs)
        if depth_obs is not None:
            transition.depth_obs = depth_obs.clone()  # shape: [B, C, H, W]
            actor_features = self._actor_critic.feature_extractor(depth_obs)
        else:
            transition.depth_obs = None
            actor_features = actor_obs

        if rgb_obs is not None:
            transition.rgb_obs = rgb_obs.clone()
            actor_features = self._actor_critic.feature_extractor_rgb(rgb_obs)
        else:
            transition.rgb_obs = None

        # Policy forward pass
        transition.actions = self._actor_critic.act(actor_features).detach()
        transition.values = self._actor_critic.get_value(critic_obs).detach()
        transition.actions_log_prob = self._actor_critic.get_actions_log_prob(
            transition.actions
        ).detach()
        transition.action_mean = self._actor_critic.action_mean.detach()
        transition.action_sigma = self._actor_critic.action_std.detach()

        # Store obs
        transition.actor_obs = actor_obs
        transition.critic_obs = critic_obs

        return transition

    def _update_transition(self, rewards, dones, infos, transition: PPOTransition):
        transition.rewards = rewards.clone()  # shape: [B,]
        transition.dones = dones  # shape: [B,]

        # Handle timeout bootstrapping
        if "time_outs" in infos:
            transition.rewards += self._cfg.algo.gae_gamma * (
                transition.values[:, 0] * infos["time_outs"]
            )

        # Store and reset
        self._rollouts.append(transition)
        transition.clear()
        self._actor_critic.reset(dones)

    def update(self):
        """Update policy using the collected experience."""
        total_pg_loss = []
        total_value_loss = []
        total_entropy = []
        if self._actor_critic.is_recurrent:
            generator = self._rollouts.recurrent_minibatch_gen(
                num_mini_batches=self._cfg.algo.num_mini_batches,
                num_epochs=self._cfg.algo.num_epochs,
            )
        else:
            generator = self._rollouts.minibatch_gen(
                num_mini_batches=self._cfg.algo.num_mini_batches,
                num_epochs=self._cfg.algo.num_epochs,
            )
        #
        for batch_dict in generator:
            old_log_probs = torch.squeeze(batch_dict["action_logprobs"])
            #
            actor_features = batch_dict["actor_obs"]
            if batch_dict["depth_obs"] is not None:
                # TODO: Hack solution for depth observation
                B, T, C, H, W = batch_dict["depth_obs"].shape
                depth = batch_dict["depth_obs"].reshape(-1, C, H, W)
                actor_features = self._actor_critic.feature_extractor(depth).reshape(B, T, -1)

            self._actor_critic.act(
                actor_features,
                masks=batch_dict["masks"],
                hidden_states=batch_dict["actor_hidden"],
            )
            new_log_probs = self._actor_critic.get_actions_log_prob(batch_dict["actions"])
            batch_adv = batch_dict["advantages"]
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = -torch.squeeze(batch_adv) * ratio
            surr2 = -torch.squeeze(batch_adv) * torch.clamp(
                ratio, 1.0 - self._cfg.algo.clip_param, 1.0 + self._cfg.algo.clip_param
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
            values = self._actor_critic.get_value(
                batch_dict["critic_obs"],
                masks=batch_dict["masks"],
                hidden_states=batch_dict["critic_hidden"],
            )
            if self._cfg.algo.clip_value_loss:
                value_clipped = batch_dict["values"] + (values - batch_dict["values"]).clamp(
                    -self._cfg.algo.clip_param, self._cfg.algo.clip_param
                )
                value_losses = (values - batch_dict["returns"]).pow(2)
                value_losses_clipped = (value_clipped - batch_dict["returns"]).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch_dict["returns"] - values).pow(2).mean()

            # Calculate entropy loss
            entropy = self._actor_critic.entropy
            entropy_loss = entropy.mean()

            # Total loss
            total_loss = (
                policy_loss
                + self.cfg.algo.value_coef * value_loss
                - self.cfg.algo.ent_coef * entropy_loss
            )

            # Optimization step
            self._actor_optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self.cfg.algo.max_grad_norm)
            self._actor_optimizer.step()

            # Log losses
            total_pg_loss += [policy_loss.item()]
            total_value_loss += [value_loss.item()]
            total_entropy += [entropy_loss.item()]

        # reset rollouts buffer
        self._rollouts.reset()

        # Log average losses
        return np.mean(total_pg_loss), np.mean(total_value_loss), np.mean(total_entropy)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self._actor_critic.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
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
        self._actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self._actor_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["iter"]
        if self.cfg.norm_obs:
            self.actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer"])
            self.critic_obs_normalizer.load_state_dict(checkpoint["critic_obs_normalizer"])
        return checkpoint.get("infos", None)

    def train_mode(self):
        self._actor_critic.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()

    def eval_mode(self):
        self._actor_critic.eval()
        self.actor_obs_normalizer.eval()
        self.critic_obs_normalizer.eval()

    def get_inference_policy(self, device=None):
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor_critic.to(device)
        # policy = self._actor_critic.act_inference
        policy = self._actor_critic
        if self.cfg.policy.norm_obs:
            self.actor_obs_normalizer.to(device)
            return policy, self.actor_obs_normalizer
        else:
            return policy, None
