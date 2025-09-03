import torch
import torch.nn as nn

#
from gs_agent.utils.torch_utils import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] = [64, 64],
        activation: str | type[nn.Module] | nn.Module = nn.ReLU,
    ):
        """
        A fully-connected MLP.

        Args:
            input_dim:    size of the input vector
            output_dim:   size of the output vector
            hidden_layers: sizes of the hidden layers
            activation:   activation to use after each hidden layer.
                          Can be a string name, an nn.Module class, or an nn.Module instance.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            # get a fresh activation instance each time
            layers.append(get_activation(activation))
            in_dim = h

        # final layer (no activation)
        layers.append(nn.Linear(in_dim, output_dim))

        # package into a single Sequential
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RNN(nn.Module):
    def __init__(self, input_size, rnn_type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = self._unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0

    @staticmethod
    def _unpad_trajectories(trajectories, masks):
        """Does the inverse operation of  split_and_pad_trajectories()"""
        # Need to transpose before and after the masking to have proper reshaping
        return (
            trajectories.transpose(1, 0)[masks.transpose(1, 0)]
            .view(-1, trajectories.shape[0], trajectories.shape[-1])
            .transpose(1, 0)
        )


class CNN(nn.Module):
    def __init__(self, input_channels=3, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),  # [B, 32, H/4, W/4]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, H/8, W/8]
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, H/8, W/8]
            nn.ReLU(),
        )
        # TODO: Adjust the output dimension based on the input image size
        self.fc = nn.Linear(64 * 8 * 8, output_dim)  # Assuming image size ~64x64

    def forward(self, x):
        """
        x - [B, C, H, W] - batch of images
        """
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        return self.fc(x)  # [B, output_dim]


class VisionGRUPolicy(nn.Module):
    def __init__(
        self, input_channels=3, cnn_output_dim=128, gru_hidden_dim=256, action_dim=6, state_dim=None
    ):
        """
        Args:
            input_channels: Number of input channels (e.g., 3 for RGB images)
            cnn_output_dim: Output dimension of the CNN feature extractor
            gru_hidden_dim: Hidden dimension of the GRU
            action_dim: Dimension of the output actions
            state_dim: Optional state dimension (not used in this implementation)
        """
        super().__init__()
        self.cnn = CNN(input_channels, output_dim=cnn_output_dim)
        gru_input_size = cnn_output_dim if state_dim is None else state_dim + cnn_output_dim
        self.gru = nn.GRU(input_size=gru_input_size, hidden_size=gru_hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, 128), nn.ReLU(), nn.Linear(128, action_dim)
        )
        self._gru_hidden = None  # Optional hidden state storage (if used)

    def forward(self, images, hidden_state=None, state=None):
        """
        Args:
            images: [B, T, C, H, W]  - batch of image sequences
            hidden_state: [1, B, H]  - optional GRU hidden state

        Returns:
            actions: [B, T, action_dim]
            new_hidden: [1, B, H]
        """
        B, T, C, H, W = images.shape
        images = images.view(B * T, C, H, W)
        features = self.cnn(images)  # [B*T, cnn_output_dim]
        features = features.view(B, T, -1)  # [B, T, cnn_output_dim]

        if state is not None:
            # If state is provided, concatenate it with the features
            features = torch.cat((features, state.unsqueeze(1).expand(-1, T, -1)), dim=-1)

        gru_out, new_hidden = self.gru(features, hidden_state)  # [B, T, H]
        actions = self.mlp(gru_out)  # [B, T, action_dim]
        return actions, new_hidden
