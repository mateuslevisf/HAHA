import torch.nn as nn


class MLPBase(nn.Module):
    """
    MLP Base network for actors and critic
    """
    def __init__(self, input_dim, hidden_size=64, use_orthogonal=True):
        super(MLPBase, self).__init__()
        self.hidden_size = hidden_size

        # Initialization function
        init_method = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        gain = 2 ** 0.5

        def init_(m):
            if isinstance(m, nn.Linear):
                init_method(m.weight, gain)
                nn.init.constant_(m.bias, 0)
            return m

        self.net = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)