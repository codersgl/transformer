import torch
import torch.nn as nn
from transformer.model.attention import MultiHeadAttention
from transformer.model.misc import FeedForwardNet


class EncoderBlock(nn.Module):
    def __init__(
        self, embed_dim, mid_dim, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward_net = FeedForwardNet(embed_dim, mid_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask=None):
        x = x + self.dropout1(self.multi_head_attention(self.norm1(x), mask=pad_mask))
        x = x + self.dropout2(self.feed_forward_net(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        mid_dim,
        num_heads: int = 8,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(embed_dim, mid_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, pad_mask=pad_mask)
        x = self.norm(x)
        return x
