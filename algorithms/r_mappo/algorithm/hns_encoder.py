import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import torch.utils.checkpoint as cp
from algorithms.utils.util import init
from algorithms.utils.running_normalization import RunningNormalization


def masked_avg_pooling(scores, mask=None):
    if mask is None:
        return scores.mean(-2)
    else:
        assert mask.shape[-1] == scores.shape[-2]
        masked_scores = scores * mask.unsqueeze(-1)
        return masked_scores.sum(-2) / (mask.sum(-1, keepdim=True) + 1e-5)


class HNSEncoder(nn.Module):
    # special design for reproducing hide-and-seek paper
    def __init__(self, obs_space, omniscient, use_orthogonal):
        super(HNSEncoder, self).__init__()
        self.omniscient = omniscient
        self.self_obs_keys = ['observation_self', 'lidar']
        self.ordered_other_obs_keys = ['agent_qpos_qvel', 'box_obs', 'ramp_obs']
        self.ordered_obs_mask_keys = ['mask_aa_obs', 'mask_ab_obs', 'mask_ar_obs']

        self_dim = obs_space['observation_self'][-1] + obs_space['lidar'][-1]
        others_shape_dict = deepcopy(obs_space)
        others_shape_dict.pop('observation_self')
        others_shape_dict.pop('lidar')

        self.lidar_conv = nn.Conv1d(1, 1, 3, padding=1, padding_mode='circular')
        self.embedding_layer = CatSelfEmbedding(self_dim, others_shape_dict, 128, use_orthogonal=use_orthogonal)
        self.attn = ResidualMultiHeadSelfAttention(128, 4, 32, use_orthogonal=use_orthogonal)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=nn.init.calculate_gain('relu'))

        self.dense = nn.Sequential(init_(nn.Linear(256, 256)), nn.ReLU(inplace=True), nn.LayerNorm(256))

    def forward(self, inputs, use_ckpt=False):
        lidar = inputs['lidar']
        if len(lidar.shape) == 4:
            lidar = lidar.view(-1, *lidar.shape[2:])
        x_lidar = self.lidar_conv(lidar).reshape(*inputs['lidar'].shape[:-2], -1)
        x_self = torch.cat([inputs['observation_self'], x_lidar], dim=-1)

        x_other = {k: inputs[k] for k in self.ordered_other_obs_keys}
        x_self, x_other = self.embedding_layer(x_self, **x_other)

        if self.omniscient:
            mask = torch.cat([inputs[k + '_spoof'] for k in self.ordered_obs_mask_keys], -1)
        else:
            mask = torch.cat([inputs[k] for k in self.ordered_obs_mask_keys], -1)
        if use_ckpt:
            pooled_attn_other = cp.checkpoint(lambda x, y, z: masked_avg_pooling(self.attn(x, y, z), y), x_other, mask, use_ckpt)
        else:
            attn_other = self.attn(x_other, mask, use_ckpt)
            pooled_attn_other = masked_avg_pooling(attn_other, mask)
        x = torch.cat([x_self, pooled_attn_other], dim=-1)
        return self.dense(x)


class CatSelfEmbedding(nn.Module):
    def __init__(self, self_dim, others_shape_dict, d_embedding, use_orthogonal=True):
        super(CatSelfEmbedding, self).__init__()
        self.self_dim = self_dim
        self.others_shape_dict = others_shape_dict
        self.d_embedding = d_embedding

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=nn.init.calculate_gain('relu'))

        def get_layer(input_dim, output_dim):
            return nn.Sequential(init_(nn.Linear(input_dim, output_dim)), nn.ReLU(inplace=True), nn.LayerNorm(output_dim))

        self.others_keys = sorted(self.others_shape_dict.keys())
        self.self_embedding = get_layer(self_dim, d_embedding)
        for k in self.others_keys:
            if 'mask' not in k:
                setattr(self, k + '_fc', get_layer(others_shape_dict[k][-1] + self_dim, d_embedding))

    def forward(self, self_vec, **inputs):
        other_embeddings = []
        self_embedding = self.self_embedding(self_vec)
        self_vec_ = self_vec.unsqueeze(-2)
        for k, x in inputs.items():
            assert k in self.others_keys and 'mask' not in k
            expand_shape = [-1 for _ in range(len(x.shape))]
            expand_shape[-2] = x.shape[-2]
            x_ = torch.cat([self_vec_.expand(*expand_shape), x], -1)
            other_embeddings.append(getattr(self, k + '_fc')(x_))

        other_embeddings = torch.cat(other_embeddings, dim=-2)
        return self_embedding, other_embeddings


def ScaledDotProductAttention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(-2).unsqueeze(-2)
        scores = scores - (1 - mask) * 1e10
    # in case of overflow
    scores = scores - scores.max(dim=-1, keepdim=True)[0]
    scores = F.softmax(scores, dim=-1)
    if mask is not None:
        # for stablity
        scores = scores * mask

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, heads, d_head, dropout=0.0, use_orthogonal=True):
        super(MultiHeadSelfAttention, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.d_model = d_head * heads
        self.d_head = d_head
        self.h = heads

        self.q_linear = init_(nn.Linear(input_dim, self.d_model))
        self.v_linear = init_(nn.Linear(input_dim, self.d_model))
        self.k_linear = init_(nn.Linear(input_dim, self.d_model))
        # self.attn_dropout = nn.Dropout(dropout)
        self.attn_dropout = None

    def forward(self, x, mask, use_ckpt=False):
        # perform linear operation and split into h heads
        k = self.k_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)
        q = self.q_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)
        v = self.v_linear(x).view(*x.shape[:-1], self.h, self.d_head).transpose(-2, -3)

        # calculate attention
        scores = ScaledDotProductAttention(q, k, v, self.d_head, mask, self.attn_dropout)

        # concatenate heads and put through final linear layer
        return scores.transpose(-2, -3).contiguous().view(*x.shape[:-1], self.d_model)


class ResidualMultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, heads, d_head, dropout=0.0, use_orthogonal=True):
        super(ResidualMultiHeadSelfAttention, self).__init__()
        self.d_model = heads * d_head
        self.attn = MultiHeadSelfAttention(input_dim, heads, d_head, dropout, use_orthogonal)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=nn.init.calculate_gain('relu'))

        self.dense = nn.Sequential(init_(nn.Linear(self.d_model, self.d_model)), nn.ReLU(inplace=True))
        self.residual_norm = nn.LayerNorm(self.d_model)
        # self.dropout_after_attn = nn.Dropout(dropout)
        self.dropout_after_attn = None

    def forward(self, x, mask, use_ckpt=False):
        scores = self.dense(self.attn(x, mask, use_ckpt))
        if self.dropout_after_attn is not None:
            scores = self.dropout_after_attn(scores)
        return self.residual_norm(x + scores)
