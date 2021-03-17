import torch.nn as nn
from .util import init
"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh, nn.ReLU][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = []
        layers += [init_(nn.Linear(input_dim, hidden_size)), active_func(), nn.LayerNorm(hidden_size)]
        for j in range(layer_N):
            layers += [init_(nn.Linear(hidden_size, hidden_size)), active_func(), nn.LayerNorm(hidden_size)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = _use_orthogonal = args.use_orthogonal
        self._use_ReLU = _use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = _layer_N = args.layer_N
        self.hidden_size = hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, hidden_size, _layer_N, _use_orthogonal, _use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x
