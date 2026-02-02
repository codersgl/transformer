import torch
import torch.nn as nn
from transformer.model.attention import MultiHeadAttention
from transformer.model.misc import FeedForwardNet


class DecoderBlock(nn.Module):
    def __init__(
        self, embed_dim, mid_dim, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.masked_multi_head_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.multi_head_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward_net = FeedForwardNet(embed_dim, mid_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
    ):
        x = x + self.dropout(self.masked_multi_head_attn(self.norm1(x), mask=tgt_mask))
        x = x + self.dropout(
            self.multi_head_attn(
                query=self.norm2(x),
                key=encoder_output,
                value=encoder_output,
                mask=src_mask,
            )
        )
        x = x + self.dropout(self.feed_forward_net(self.norm3(x)))
        return x


class Decoder(nn.Module):
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
                DecoderBlock(embed_dim, mid_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask=None,
        tgt_mask=None,
    ):
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
        x = self.norm(x)
        return x
