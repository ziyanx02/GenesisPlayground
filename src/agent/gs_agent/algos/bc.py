import os
import statistics
import time
from collections import deque

import numpy as np
import torch
from gs_agent.buffers.bc_buffer import BCBuffer
from gs_agent.buffers.transition import BCTransition
from gs_agent.configs.bc_cfg import BCConfig
from gs_agent.modules.actor_critic import VisionActor
from gs_agent.modules.normalizer import EmpiricalNormalization
from gs_agent.utils.logger import configure as logger_configure
from gs_agent.utils.policy_loader import load_policy_from_file
from tensordict import TensorDict


class BC:
    """
    Base class for behavior cloning algorithms.
    """

    def __init__(self, env, cfg: BCConfig, device="cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device

        #
        self._state_dim = self._env.actor_obs_dim
        self._action_dim = self._env.action_dim
        self._depth_shape = self._env.depth_shape
        self._rgb_shape = self._env.rgb_shape
        #
        self._num_envs = env.num_envs
        self._num_steps = cfg.runner.num_steps_per_env
        #
        self.current_iter = 0

        #
        self._init()

    def _init(self):
        #
        # self._actor = ActorRecurrent(
        #     input_channels=1,
        #     action_dim=self._action_dim,
        #     state_dim=self._state_dim,
        #     cnn_output_dim=self._cfg.policy.cnn_output_dim,
        #     rnn_hidden_dim=self._cfg.policy.rnn_hidden_size,
        # )
        state_dim = 7 + 4
        self._actor = VisionActor(
            input_channels=1,
            action_dim=self._action_dim,
            state_dim=state_dim,
            cnn_output_dim=self._cfg.policy.cnn_output_dim,
        )

        self._rollouts = BCBuffer(
            num_envs=self._num_envs,
            max_steps=int(1e2),
            state_dim=state_dim,
            action_dim=self._action_dim,
            img_res=self._env.img_resolution,
            device=self._device,
        )

        if self._cfg.policy.norm_obs:
            print("Using Empirical Normalization!")
            self.actor_obs_normalizer = EmpiricalNormalization(
                shape=(self._state_dim,), until=1e6
            ).to(self._device)
        else:
            self.actor_obs_normalizer = torch.nn.Identity().to(self._device)  # no normalization

        # Initialize optimizer
        self._optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=self._cfg.algo.learning_rate
        )

        #
        checkpoints_path = (
            "/home/yunlong/Projects/GenesisLab/gs-core/logs/gsppo_goal_reach_20250618_102410"
        )
        self._teacher = load_policy_from_file(checkpoints_path, device=self._device)

    def train(self, num_iters: int, log_dir: str | None = None):
        self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer, T_max=num_iters, eta_min=self._cfg.algo.learning_rate * 0.1
        )

        # initialize writer
        logger = logger_configure(folder=log_dir, format_strings=["stdout", "csv", "wandb"])

        # Get initial observations
        obs, extra_infos = self._env.get_observations()

        #
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        curr_reward_sum = torch.zeros(self._env.num_envs, device=self._device, dtype=torch.float)
        curr_ep_len = torch.zeros(self._env.num_envs, device=self._device, dtype=torch.float)

        # Main training loop
        start_iter = self.current_iter
        start_train_time = time.time()
        for it in range(start_iter, self.current_iter + num_iters):
            reward_dict = TensorDict(
                {},
                device=self._device,
                batch_size=[self._num_steps, self._num_envs],
            )
            self.train_mode()  # switch to train mode (for dropout for example)
            transition = BCTransition()
            with torch.inference_mode():
                # collect rollouts and compute returns & advantages
                start = time.time()
                for step in range(self._num_steps):
                    #
                    depth_obs = self._env.get_depth_image(normalize=True)
                    transition.depth_obs = depth_obs.clone()

                    # teacher action
                    teacher_act = self._teacher.act(obs).detach()
                    transition.actions = teacher_act.clone()

                    # student act
                    features = self._actor.feature_extractor(depth_obs)
                    ee_pose, obj_ori = self._env.get_ee_pose(), self._env.get_obj_ori()
                    state_obs = torch.cat([ee_pose, obj_ori], dim=-1)
                    features = torch.cat([features, state_obs], dim=-1)
                    student_act = self._actor.act(features).detach()

                    transition.state_obs = state_obs
                    #
                    if self._actor.is_recurrent:
                        # Initialize hidden states for recurrent actor-critic
                        transition.actor_hidden = self._actor.get_hidden_states()

                    # Step environment
                    next_obs, reward, done, extra_infos = self._env.step(student_act)
                    reward_dict[step] = extra_infos["reward_dict"]

                    transition.dones = done

                    #
                    self._rollouts.append(transition)
                    self._actor.reset(done)
                    transition.clear()

                    obs = next_obs

                    if logger is not None:
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

            start = time.time()
            # Update policy by optimizing PPO objective
            fps = self._num_steps * self._num_envs / collection_time
            mean_loss = self.update()
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
            logger.record("train/mse_loss", mean_loss)
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

    def update(self):
        """Update policy using the collected experience."""
        total_loss = []
        if self._actor.is_recurrent:
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
        #
        for batch_dict in generator:
            # TODO: Hack solution for depth observation
            features = self._actor.feature_extractor(batch_dict["depth_obs"])
            state = batch_dict["state_obs"]
            features = torch.cat([features, state], dim=-1)
            student_actions = self._actor.act(features)
            #
            loss = (batch_dict["actions"] - student_actions).pow(2).mean()

            # Optimization step
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._cfg.algo.max_grad_norm)
            self._optimizer.step()

            # Log losses
            total_loss += [loss.item()]

        # # reset rollouts buffer
        # self._rollouts.reset()

        # Log average losses
        return np.mean(total_loss)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self._actor.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "iter": self.current_iter,
        }
        if self._cfg.policy.norm_obs:
            saved_dict["actor_obs_normalizer"] = self.actor_obs_normalizer.state_dict()
        if infos is not None:
            saved_dict.update(infos)
        torch.save(saved_dict, path)

    def load(self, path, load_optimizer=True):
        checkpoint = torch.load(path)
        self._actor.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["iter"]
        if self._cfg.policy.norm_obs:
            self.actor_obs_normalizer.load_state_dict(checkpoint["actor_obs_normalizer"])
        return checkpoint.get("infos", None)

    def train_mode(self):
        self._actor.train()
        self.actor_obs_normalizer.train()

    def eval_mode(self):
        self._actor.eval()
        self.actor_obs_normalizer.eval()

    def get_inference_policy(self, device=None):
        """Get the inference policy for evaluation."""
        self.eval_mode()
        if device is not None:
            self._actor.to(device)
        if self._cfg.policy.norm_obs:
            self.actor_obs_normalizer.to(device)
            return self._actor, self.actor_obs_normalizer
        else:
            return self._actor, None
