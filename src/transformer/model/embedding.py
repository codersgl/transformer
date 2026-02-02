import torch
import math
import torch.nn as nn


class PositionalEncoder(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)  # [max_len, embed_dim]

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor):
        """
        Args:
            x(torch.Tensor): [batch_size, seq_len, embed_dim]
        """
        return x + self.pe[:, : x.size(1), :]
