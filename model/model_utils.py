import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn import TransformerEncoderLayer, TransformerEncoder

import math
from config import device, config
from data.data_utils import pitch_num_to_id, id_to_pitch_num

class SpiralEmbedding4d(nn.Module):
    def __init__(self, discount_fact=32, z_bias=-1, min_pitch=55-6, max_pitch=88+5):
        super(SpiralEmbedding4d, self).__init__()
        self.discount_fact = discount_fact
        self.z_bias = z_bias
        self.d_squared_to_rest = max(
            (max_pitch * 1.0 / self.discount_fact + self.z_bias) ** 2,
            (min_pitch * 1.0 / self.discount_fact + self.z_bias) ** 2,
        )

    # p is a tensor of integer in size (*)
    # output an embedding in size (*,4)
    def forward(self, p):
        non_rest = p >= 0
        rest = p < 0
        x = p.clone().to(torch.float32)
        y = p.clone().to(torch.float32)
        z = p.clone().to(torch.float32)
        u = p.clone().to(torch.float32)
        x[non_rest] = torch.cos(2 * math.pi * p[non_rest] / 12)
        y[non_rest] = torch.sin(2 * math.pi * p[non_rest] / 12)
        z[non_rest] = p[non_rest] * 1.0 / self.discount_fact + self.z_bias
        u[non_rest] = torch.sqrt(-z[non_rest] * z[non_rest] + self.d_squared_to_rest)
        x[rest] = x[rest] * 0
        y[rest] = y[rest] * 0
        z[rest] = z[rest] * 0
        u[rest] = u[rest] * 0
        return torch.stack([x, y, z, u], dim=-1)

class SpiralEmbeddingChord(nn.Module):
    def __init__(self):
        super(SpiralEmbeddingChord, self).__init__()
        self.emb_tensor = torch.ones([12,2])
        for i in range(12):
            self.emb_tensor[i] = torch.FloatTensor([math.cos(2*math.pi*i/12), math.sin(2*math.pi*i/12)]).to(device)
    # c is a mutli-hot chord vector in size (*,12)
    # output an embedding in size (*,12,2)
    def forward(self, c):
        c_tmp = torch.stack([c,c],dim=-1) # dim: (*,12,2)
        return c_tmp * self.emb_tensor
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)[:, : pe[:, 0, 1::2].shape[1]]
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)