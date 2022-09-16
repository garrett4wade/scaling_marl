import torch
import torch.nn as nn
"""RNN modules."""


class RNNLayer(nn.Module):
    def __init__(self, inputs_dim, outputs_dim, rec_n, use_orthogonal):
        super(RNNLayer, self).__init__()
        self._rec_n = rec_n
        self._use_orthogonal = use_orthogonal
        self.hidden_size = inputs_dim

        self.rnn = nn.LSTM(inputs_dim, outputs_dim, num_layers=self._rec_n)
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                if self._use_orthogonal:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
        self.norm = nn.LayerNorm(outputs_dim)

    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            # rollout
            print('self.hidden_size', self.hidden_size)
            print('x', x.shape, 'hxs', hxs.shape, 'masks', masks.shape)
            h, c = (hxs * masks.unsqueeze(-1)).transpose(0, 1).split(self.hidden_size, -1)
            print('h', h.shape, 'c', c.shape)
            x, h_and_c = self.rnn(x.unsqueeze(0), (h.contiguous(), c.contiguous()))
            x = x.squeeze(0)
            hxs = torch.cat(h_and_c, -1).transpose(0, 1)
        else:
            has_zeros = (masks[1:] == 0.0).any(dim=1).nonzero(as_tuple=True)[0].cpu().numpy()
            has_zeros = [0] + (has_zeros + 1).tolist() + [x.shape[0]]

            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]
                h, c = (hxs * masks[start_idx].view(1, -1, 1)).split(self.hidden_size, -1)
                rnn_scores, h_and_c = self.rnn(x[start_idx:end_idx], (h.contiguous(), c.contiguous()))
                hxs = torch.cat(h_and_c, -1)
                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)

        x = self.norm(x)
        return x, hxs
