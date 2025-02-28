import torch as th
import torch.nn as nn
from marl.mappo.mlp_base import MLPBase

class ActorNetwork(nn.Module):
    """
    Actor network for MAPPO that outputs subtask selection probabilities
    """
    def __init__(self, obs_dim, action_dim, hidden_size=64, use_orthogonal=True):
        super(ActorNetwork, self).__init__()
        self.base = MLPBase(obs_dim, hidden_size, use_orthogonal)

        # Initialization function for output layer
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
            return m

        self.action_head = init_(nn.Linear(hidden_size, action_dim))

    def forward(self, obs, action_masks=None):
        features = self.base(obs)
        action_logits = self.action_head(features)

        # Apply action mask if provided - action mask should be tensors
        if action_masks is not None:
            # Ensure dimensions match
            if action_masks.dim() < action_logits.dim():
                action_masks = action_masks.unsqueeze(0)
            action_logits[~action_masks] = -1e10

        return th.distributions.Categorical(logits=action_logits)