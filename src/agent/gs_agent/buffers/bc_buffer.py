import torch
from tensordict import TensorDict

from gs_agent.buffers import base_buffer
from gs_agent.buffers.transition import BCTransition
from gs_agent.utils.misc import split_and_pad_trajectories

STATE_OBS = "state_obs"
ACTIONS = "actions"
DONES = "dones"
RGB_OBS = "rgb_obs"
DEPTH_OBS = "depth_obs"
ACTOR_HIDDEN = "actor_hidden"
MASKS = "masks"


class BCBuffer(base_buffer.BaseBuffer):
    """
    A fixed-size buffer for storing rollouts for multi-env on-policy training,
    with GAE/Generalized Advantage Estimation.
    """

    def __init__(
        self,
        num_envs: int,
        max_steps: int,
        state_dim: int,
        action_dim: int,
        img_res: tuple[int, int] | None = None,
        device: str = "cpu",
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
        self._num_envs = num_envs
        self._max_steps = max_steps
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device

        # Track current size dynamically
        self._idx = 0
        self._size = 0

        self._actor_hidden = None  # type: ignore

        ## RGB image shape (channels, height, width)
        self._rgb_shape = (3, img_res[0], img_res[1]) if img_res is not None else None
        self._depth_shape = (1, img_res[0], img_res[1]) if img_res is not None else None

        # Initialize buffer
        self._buffer = self._init_buffers(state_dim, action_dim)

    def _init_buffers(self, state_dim: int, action_dim: int) -> TensorDict:
        max_steps, num_envs = self._max_steps, self._num_envs
        buffer = TensorDict(
            {
                STATE_OBS: torch.zeros(max_steps, num_envs, state_dim),
                ACTIONS: torch.zeros(max_steps, num_envs, action_dim),
                DONES: torch.zeros(max_steps, num_envs, 1).byte(),
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
        self._size = 0
        self._actor_hidden = None

    def append(self, transition: BCTransition) -> None:
        idx = self._idx % self._max_steps

        #
        self._buffer[STATE_OBS][idx].copy_(transition.state_obs)
        self._buffer[ACTIONS][idx].copy_(transition.actions)
        self._buffer[DONES][idx].copy_(transition.dones.unsqueeze(-1).byte())
        #
        self._append_hidden_states(transition.actor_hidden, idx)
        # Append RGB and depth images if available
        if self._rgb_shape is not None and transition.rgb_obs is not None:
            self._buffer[RGB_OBS][idx].copy_(transition.rgb_obs)
        if self._depth_shape is not None and transition.depth_obs is not None:
            self._buffer[DEPTH_OBS][idx].copy_(transition.depth_obs)

        # Increment index
        self._idx += 1
        self._size = min(self._size + 1, self._max_steps)

    def _append_hidden_states(self, hidden_state, idx):
        # actor
        if hidden_state is not None:
            # make a tuple out of GRU hidden state sto match the LSTM format
            hid_a = hidden_state if isinstance(hidden_state, tuple) else (hidden_state,)
            # initialize if needed
            if self._actor_hidden is None and hid_a is not None:
                self._actor_hidden = [
                    torch.zeros(
                        self._buffer[ACTIONS].shape[0],
                        *hid_a[i].shape,
                        device=self._device,
                    )
                    for i in range(len(hid_a))
                ]
            # copy the states
            for i in range(len(hid_a)):
                self._actor_hidden[i][idx].copy_(hid_a[i])

    def is_full(self):
        return self._idx >= self._max_steps

    def __len__(self):
        return self._idx * self._num_envs

    def minibatch_gen(self, num_mini_batches: int, num_epochs: int):
        buffer_size = self._size * self._num_envs
        indices = torch.randperm(buffer_size)
        # Calculate the size of each mini-batch
        batch_size = buffer_size // num_mini_batches
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
                yield batch

    def recurrent_minibatch_gen(self, num_mini_batches: int, num_epochs: int):
        """
        Yield batches of full sequences (timesteps) with corresponding hidden states.
        Each "mini-batch" is composed of full episodes or padded ones.
        """
        padded_state_trajectories, trajectory_masks = split_and_pad_trajectories(
            self._buffer[STATE_OBS], self._buffer[DONES]
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

        #
        minibatch_size = self._size // num_mini_batches

        for _ in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start, end = i * minibatch_size, (i + 1) * minibatch_size
                #
                dones = self._buffer[DONES].squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:end])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                state_obs_batch = padded_state_trajectories[:, first_traj:last_traj]
                depth_obs_batch = None
                if padded_depth_trajectories is not None:
                    # if depth is available, slice it as well
                    depth_obs_batch = padded_depth_trajectories[:, first_traj:last_traj]

                batch = {
                    STATE_OBS: state_obs_batch,
                    DEPTH_OBS: depth_obs_batch,
                    ACTIONS: self._buffer[ACTIONS][:, first_traj:last_traj],
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
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch

                batch[MASKS] = masks_batch
                batch[ACTOR_HIDDEN] = hid_a_batch
                yield batch
                first_traj = last_traj
