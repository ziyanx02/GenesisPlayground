import random
import statistics
from collections import deque

import cv2
import numpy as np
import torch
from tensordict import TensorDict


def validate_tensordict(
    td: TensorDict, spec: dict[str, tuple[type, tuple]], prefix: str = ""
) -> None:
    for key, (expected_type, expected_shape) in spec.items():
        if key not in td.keys():
            raise KeyError(f"Missing key '{prefix + key}' in TensorDict")
        val = td[key]
        if not isinstance(val, expected_type):
            raise TypeError(f"Key '{prefix + key}' expected {expected_type}, got {type(val)}")
        if expected_shape != (None,) and val.shape[-len(expected_shape) :] != expected_shape:
            raise ValueError(
                f"Key '{prefix + key}' expected shape ending in {expected_shape}, got {tuple(val.shape)}"
            )


def maybe_validate_tensordict(
    td: TensorDict, spec: dict[str, tuple[type, tuple]], prefix: str = "", DEBUG_MODE: bool = False
) -> None:
    if DEBUG_MODE:
        validate_tensordict(td, spec, prefix)


def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat(
        (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0])
    )
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (
        torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),
    )
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(
        0, tensor.shape[0], device=tensor.device
    ).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EpisodeStats:
    """Tracks episode-level statistics for PPO training."""

    def __init__(self, maxlen: int, num_envs: int, device: torch.device):
        self.ep_infos = []
        self.rewbuffer = deque(maxlen=maxlen)
        self.lenbuffer = deque(maxlen=maxlen)
        self.cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=device)
        self.device = device

    def update(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        """Update episode statistics with new transition data."""
        if "episode" in infos:
            self.ep_infos.extend(infos["episode"])

        # Update running episode stats
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        # Handle completed episodes
        new_ids = (dones > 0).nonzero(as_tuple=False)
        self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        self.cur_reward_sum[new_ids] = 0
        self.cur_episode_length[new_ids] = 0

    @property
    def stats_dict(self):
        """Get current episode statistics as a dictionary."""
        if len(self.rewbuffer) == 0:
            return {}
        return {
            "reward_mean": statistics.mean(self.rewbuffer),
            "length_mean": statistics.mean(self.lenbuffer),
        }


def save_images_to_video(images, output_path, fps=30):
    """
    Save a list of images (numpy arrays) as a video.

    Args:
        images (List[np.ndarray]): List of images (H, W, 3) in uint8 format.
        output_path (str): Output path of the video (e.g., 'video.mp4').
        fps (int): Frames per second.
    """
    if len(images) == 0:
        raise ValueError("No images provided for video saving.")

    height, width, _ = images[0].shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    for img in images:
        writer.write(img)

    writer.release()
    print(f"[INFO] Video saved to: {output_path}")
