import torch
from tensordict import TensorDict

from gs_agent.bases.buffer import BaseBuffer
from gs_agent.buffers.transition import PPOTransition
from gs_agent.utils.misc import split_and_pad_trajectories

ACTOR_OBS = "actor_obs"
CRITIC_OBS = "critic_obs"
ACTIONS = "actions"
REWARDS = "rewards"
DONES = "dones"
VALUES = "values"
ACTION_LOGPROBS = "action_logprobs"
ACTION_MEAN = "action_mean"
ACTION_SIGMA = "action_sigma"
ADVANTAGES = "advantages"
RETURNS = "returns"
RGB_OBS = "rgb_obs"
DEPTH_OBS = "depth_obs"
ACTOR_HIDDEN = "actor_hidden"
CRITIC_HIDDEN = "critic_hidden"
MASKS = "masks"


class GAEBuffer(BaseBuffer):
    """
    A fixed-size buffer for storing rollouts for multi-env on-policy training,
    with GAE/Generalized Advantage Estimation.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        actor_obs_size: int,
        critic_obs_size: int,
        action_size: int,
        img_res: tuple[int, int] | None = None,
        device: torch.device = torch.device("cpu"),
        gae_gamma=0.98,
        gae_lam=0.95,
    ) -> None:
        """
        Args:
            num_envs (int): Number of parallel environments.
            max_steps (int): Maximum number of steps to store in the buffer.
            actor_obs_size (int): Dimension of the actor's observation space.
            critic_obs_size (int): Dimension of the critic's observation space.
            action_size (int): Dimension of the action space.
            device (torch.device): Device to store the buffer on (default: CPU).
            gae_gamma (float): Discount factor for GAE.
            gae_lam (float): Lambda parameter for GAE.
        """
        super().__init__()
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._actor_obs_dim = actor_obs_size
        self._critic_obs_dim = critic_obs_size
        self._action_dim = action_size
        self._device = device

        # GAE parameters (configurable)
        self._gae_gamma = gae_gamma
        self._gae_lam = gae_lam

        self._normalize_adv = True  # Whether to normalize advantages

        # Track current size dynamically
        self._idx, self._idx_hidden = 0, 0

        self._actor_hidden = None  # type: ignore
        self._critic_hidden = None  # type: ignore

        ## RGB image shape (channels, height, width)
        self._rgb_shape = (3, img_res[0], img_res[1]) if img_res is not None else None
        self._depth_shape = (1, img_res[0], img_res[1]) if img_res is not None else None

        # Initialize buffer
        self._buffer = self._init_buffers(actor_obs_size, critic_obs_size, action_size)

    def _init_buffers(self, actor_obs_dim: int, critic_obs_dim: int, action_dim: int) -> TensorDict:
        max_steps, num_envs = self._max_steps, self._num_envs
        buffer = TensorDict(
            {
                ACTOR_OBS: torch.zeros(max_steps, num_envs, actor_obs_dim),
                CRITIC_OBS: torch.zeros(max_steps, num_envs, critic_obs_dim),
                ACTIONS: torch.zeros(max_steps, num_envs, action_dim),
                REWARDS: torch.zeros(max_steps, num_envs, 1),
                DONES: torch.zeros(max_steps, num_envs, 1).byte(),
                VALUES: torch.zeros(max_steps, num_envs, 1),
                ACTION_LOGPROBS: torch.zeros(max_steps, num_envs, 1),
                ACTION_MEAN: torch.zeros(max_steps, num_envs, action_dim),
                ACTION_SIGMA: torch.zeros(max_steps, num_envs, action_dim),
                ADVANTAGES: torch.zeros(max_steps, num_envs, 1),
                RETURNS: torch.zeros(max_steps, num_envs, 1),
            },
            batch_size=[self._max_steps, self._num_envs],
            device=self._device,
        )
        return self._extend_modalities(buffer)

    def _extend_modalities(self, buffer: TensorDict) -> TensorDict:
        """
        Extend the buffer to accommodate RGB, depth images, and maybe other modalities.
        """
        if self._rgb_shape is not None and len(self._rgb_shape) != 3:
            raise ValueError(
                f"Invalid RGB image shape: {self._rgb_shape}. Expected (channels, height, width)."
            )
        if self._depth_shape is not None and len(self._depth_shape) != 3:
            raise ValueError(
                f"Invalid depth image shape: {self._depth_shape}. Expected (channels, height, width)."
            )
        #
        if self._rgb_shape is not None:
            buffer.set(RGB_OBS, torch.zeros(self._max_steps, self._num_envs, *self._rgb_shape))
        if self._depth_shape is not None:
            buffer.set(DEPTH_OBS, torch.zeros(self._max_steps, self._num_envs, *self._depth_shape))

        return buffer

    def reset(self):
        self._idx = 0
        self._actor_hidden = None
        self._critic_hidden = None

    def append(self, transition: PPOTransition) -> None:
        if self._idx >= self._max_steps:
            raise ValueError(f"Buffer full! Cannot append more than {self._max_steps} steps.")
        #
        idx = self._idx
        self._buffer[ACTOR_OBS][idx].copy_(transition.actor_obs)
        self._buffer[CRITIC_OBS][idx].copy_(transition.critic_obs)
        self._buffer[ACTIONS][idx].copy_(transition.actions)
        self._buffer[REWARDS][idx].copy_(transition.rewards.unsqueeze(-1))
        self._buffer[DONES][idx].copy_(transition.dones.unsqueeze(-1).byte())
        self._buffer[VALUES][idx].copy_(transition.values)
        self._buffer[ACTION_LOGPROBS][idx].copy_(transition.actions_log_prob.unsqueeze(-1))
        self._buffer[ACTION_MEAN][idx].copy_(transition.action_mean)
        self._buffer[ACTION_SIGMA][idx].copy_(transition.action_sigma)
        #
        self._append_hidden_states(transition.actor_hidden, transition.critic_hidden)
        # Append RGB and depth images if available
        if self._rgb_shape is not None and transition.rgb_obs is not None:
            self._buffer[RGB_OBS][idx].copy_(transition.rgb_obs)
        if self._depth_shape is not None and transition.depth_obs is not None:
            self._buffer[DEPTH_OBS][idx].copy_(transition.depth_obs)
        # Initialize hidden states
        self._buffer[ADVANTAGES][idx].zero_()  # Initialize advantages to zero
        self._buffer[RETURNS][idx].zero_()  # Initialize returns to zero
        # Increment index
        self._idx += 1

    def _append_hidden_states(self, actor_hidden_state, critic_hidden_state):
        # actor
        if actor_hidden_state is not None:
            # make a tuple out of GRU hidden state sto match the LSTM format
            hid_a = (
                actor_hidden_state
                if isinstance(actor_hidden_state, tuple)
                else (actor_hidden_state,)
            )
            # initialize if needed
            if self._actor_hidden is None and hid_a is not None:
                self._actor_hidden = [
                    torch.zeros(
                        self._buffer[ACTOR_OBS].shape[0],
                        *hid_a[i].shape,
                        device=self._device,
                    )
                    for i in range(len(hid_a))
                ]
            # copy the states
            for i in range(len(hid_a)):
                self._actor_hidden[i][self._idx].copy_(hid_a[i])

        # critic
        if critic_hidden_state is not None:
            hid_c = (
                critic_hidden_state
                if isinstance(critic_hidden_state, tuple)
                else (critic_hidden_state,)
            )

            if self._critic_hidden is None and hid_c is not None:
                self._critic_hidden = [
                    torch.zeros(
                        self._buffer[CRITIC_OBS].shape[0],
                        *hid_c[i].shape,
                        device=self._device,
                    )
                    for i in range(len(hid_c))
                ]
            # copy the states
            for i in range(len(hid_c)):
                self._critic_hidden[i][self._idx].copy_(hid_c[i])

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
                batch[MASKS] = None
                batch[ACTOR_HIDDEN] = None
                batch[CRITIC_HIDDEN] = None
                batch[DEPTH_OBS] = None
                batch[RGB_OBS] = None
                yield batch

    def recurrent_minibatch_gen(self, num_mini_batches: int, num_epochs: int):
        """
        Yield batches of full sequences (timesteps) with corresponding hidden states.
        Each "mini-batch" is composed of full episodes or padded ones.
        """
        padded_actor_obs_trajectories, trajectory_masks = split_and_pad_trajectories(
            self._buffer[ACTOR_OBS], self._buffer[DONES]
        )
        padded_critic_obs_trajectories, _ = split_and_pad_trajectories(
            self._buffer[CRITIC_OBS], self._buffer[DONES]
        )
        # TODO: Better handling depth observations if available
        padded_depth_trajectories = None
        if self._buffer.get(DEPTH_OBS) is not None:
            T, N, C, H, W = self._buffer[DEPTH_OBS].shape
            reshape_depth = self._buffer[DEPTH_OBS].reshape(T, N, C * H * W)
            padded_depth_trajectories, _ = split_and_pad_trajectories(
                reshape_depth, self._buffer[DONES]
            )
            padded_depth_trajectories = padded_depth_trajectories.reshape(T, -1, C, H, W)
        minibatch_size = self._num_envs // num_mini_batches

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * minibatch_size
                end = (i + 1) * minibatch_size

                #
                dones = self._buffer[DONES].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:end])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                actor_obs_batch = padded_actor_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                depth_obs_batch = None
                if padded_depth_trajectories is not None:
                    # if depth is available, slice it as well
                    depth_obs_batch = padded_depth_trajectories[:, first_traj:last_traj]

                batch = {
                    ACTOR_OBS: actor_obs_batch,
                    CRITIC_OBS: critic_obs_batch,
                    DEPTH_OBS: depth_obs_batch,
                    ACTIONS: self._buffer[ACTIONS][:, start:end],
                    VALUES: self._buffer[VALUES][:, start:end],
                    ADVANTAGES: self._buffer[ADVANTAGES][:, start:end],
                    RETURNS: self._buffer[RETURNS][:, start:end],
                    ACTION_LOGPROBS: self._buffer[ACTION_LOGPROBS][:, start:end],
                }
                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self._actor_hidden
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self._critic_hidden
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                batch[MASKS] = masks_batch
                batch[ACTOR_HIDDEN] = hid_a_batch
                batch[CRITIC_HIDDEN] = hid_c_batch
                yield batch
                first_traj = last_traj
