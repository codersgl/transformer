import torch
import torch.nn as nn

from transformer.model.decoder import Decoder
from transformer.model.embedding import PositionalEncoder
from transformer.model.encoder import Encoder
from typing import Optional


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        mid_dim,
        num_heads: int = 8,
        num_encoder_blocks: int = 6,
        num_decoder_blocks: int = 6,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoder = PositionalEncoder(embed_dim)
        self.encoder = Encoder(
            embed_dim, mid_dim, num_heads, num_encoder_blocks, dropout
        )
        self.decoder = Decoder(
            embed_dim, mid_dim, num_heads, num_decoder_blocks, dropout
        )
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None
    ):
        # Removed sqrt scaling for numerical stability
        # Original paper used: embedding * sqrt(d_model), but this can cause gradient explosion
        src_embed = self.embedding(src)
        src_embed = self.positional_encoder(src_embed)

        tgt_embed = self.embedding(tgt)
        tgt_embed = self.positional_encoder(tgt_embed)

        src_embed = self.dropout(src_embed)
        tgt_embed = self.dropout(tgt_embed)

        encoder_output = self.encoder(src_embed, pad_mask=src_mask)
        decoder_output = self.decoder(
            tgt_embed, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask
        )

        output = self.fc(decoder_output)
        return output
