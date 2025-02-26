import torch.nn as nn
from marl.mappo.mlp_base import MLPBase

class CriticNetwork(nn.Module):
    """
    Centralized critic network for MAPPO
    """
    def __init__(self, obs_dim, hidden_size=64, use_orthogonal=True):
        super(CriticNetwork, self).__init__()
        self.base = MLPBase(obs_dim, hidden_size, use_orthogonal)

        # Initialization function for output layer
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            return m

        self.value_head = init_(nn.Linear(hidden_size, 1))

    def forward(self, obs):
        """
        Forward pass through critic network

        Args:
            obs (th.Tensor): Centralized observation tensor

        Returns:
            th.Tensor: Value estimate
        """
        features = self.base(obs)
        return self.value_head(features)