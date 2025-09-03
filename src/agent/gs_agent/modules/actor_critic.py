import torch
import torch.nn as nn
from torch.distributions import Normal

from gs_agent.modules.models import CNN, MLP, RNN


class ActorCritic(nn.Module):
    """Actor-Critic network with separate observation spaces."""

    is_recurrent = False
    actor_hidden_size = None
    critic_hidden_size = None

    def __init__(
        self,
        actor_input_dim: int,
        critic_input_dim: int,
        act_dim: int,
        cfg,
    ):
        super().__init__()
        self.actor_obs_dim = actor_input_dim
        self.critic_obs_dim = critic_input_dim
        self.act_dim = act_dim
        self.cfg = cfg

        # Actor network
        self.actor = MLP(
            input_dim=actor_input_dim,
            output_dim=act_dim,
            hidden_layers=cfg.policy.actor_hidden,
        )

        # State-independent log standard deviation
        self.std = nn.Parameter(cfg.policy.init_noise_std * torch.ones(act_dim))
        Normal.set_default_validate_args(False)

        # Critic network
        self.critic = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_layers=cfg.policy.critic_hidden,
        )
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self, actor_obs):
        return self.actor(actor_obs)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(-1)

    def update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)

    def get_value(self, obs: torch.Tensor, **kwargs) -> torch.Tensor:
        value = self.critic(obs)
        return value


class ActorCriticRecurrent(ActorCritic):
    is_recurrent = True

    def __init__(
        self,
        actor_input_dim: int,
        critic_input_dim: int,
        act_dim: int,
        cfg,
        rnn_type: str = "lstm",
        rnn_num_layers: int = 1,
        rnn_hidden_size: int = 256,
    ):
        super().__init__(
            actor_input_dim=rnn_hidden_size,
            critic_input_dim=rnn_hidden_size,
            act_dim=act_dim,
            cfg=cfg,
        )

        self.cnn = CNN(input_channels=1, output_dim=actor_input_dim)

        #
        self.actor_hidden_size = rnn_hidden_size
        self.critic_hidden_size = rnn_hidden_size

        self.memory_a = RNN(
            input_size=actor_input_dim,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )
        self.memory_c = RNN(
            input_size=critic_input_dim,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )

        print(f"Actor CNN: {self.cnn}")
        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def feature_extractor(self, img: torch.Tensor) -> torch.Tensor:
        # CNN feature extractor
        return self.cnn(img)

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        input_a = self.memory_a(observations, masks, hidden_states)
        return super().act(input_a.squeeze(0))

    def act_inference(self, observations):
        input_a = self.memory_a(observations)
        return super().act_inference(input_a.squeeze(0))

    def get_value(self, critic_observations, masks=None, hidden_states=None):
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        return super().get_value(input_c.squeeze(0))

    def get_hidden_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.memory_a.hidden_states, self.memory_c.hidden_states


class VisionActor(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        state_dim: int,
        cnn_output_dim: int = 256,
        mlp_hidden: list[int] = [256, 256, 128],
    ):
        super().__init__()
        # depth -> cnn -> rnn -> mlp
        self.cnn = CNN(input_channels=input_channels, output_dim=cnn_output_dim)
        #
        mlp_input = cnn_output_dim + state_dim
        # Actor network
        self.mlp = MLP(
            input_dim=mlp_input,
            output_dim=action_dim,
            hidden_layers=mlp_hidden,
        )

    def feature_extractor(self, img: torch.Tensor) -> torch.Tensor:
        # CNN feature extractor
        return self.cnn(img)

    def reset(self, dones=None):
        pass

    def act(self, features, masks=None, hidden_states=None):
        return self.mlp(features)


class ActorRecurrent(nn.Module):
    is_recurrent = True

    def __init__(
        self,
        input_channels: int,
        action_dim: int,
        state_dim: int | None,
        cnn_output_dim: int = 256,
        rnn_type: str = "lstm",
        rnn_num_layers: int = 1,
        rnn_hidden_dim: int = 256,
        mlp_hidden: list[int] = [256, 256, 128],
    ):
        super().__init__()
        # depth -> cnn -> rnn -> mlp
        self.cnn = CNN(input_channels=input_channels, output_dim=cnn_output_dim)
        self.memory_a = RNN(
            input_size=cnn_output_dim,
            rnn_type=rnn_type,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_dim,
        )
        mlp_input_dim = rnn_hidden_dim  # + state_dim if state_dim is not None else rnn_hidden_dim
        # Actor network
        self.mlp = MLP(
            input_dim=mlp_input_dim,
            output_dim=action_dim,
            hidden_layers=mlp_hidden,
        )

        #
        self.actor_hidden_size = rnn_hidden_dim

    def feature_extractor(self, img: torch.Tensor) -> torch.Tensor:
        # CNN feature extractor
        return self.cnn(img)

    def reset(self, dones=None):
        self.memory_a.reset(dones)

    def act(self, img_features, masks=None, hidden_states=None):
        input_a = self.memory_a(img_features, masks, hidden_states).squeeze(0)
        return self.mlp(input_a)

    def act_inference(self, img_features):
        input_a = self.memory_a(img_features).squeeze(0)
        return self.mlp(input_a)

    def get_hidden_states(self) -> torch.Tensor:
        return self.memory_a.hidden_states
