import math

import torch
import torch.nn as nn

from transformer.model.decoder import Decoder
from transformer.model.embedding import PositionalEncoder
from transformer.model.encoder import Encoder


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
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEncoder(embed_dim)
        self.encoder = Encoder(
            embed_dim, mid_dim, num_heads, num_encoder_blocks, dropout
        )
        self.decoder = Decoder(
            embed_dim, mid_dim, num_heads, num_decoder_blocks, dropout
        )
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None, tgt_mask=None
    ):
        src_embed = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src_embed = src_embed + self.positional_encoder(src_embed)

        tgt_embed = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        tgt_embed = tgt_embed + self.positional_encoder(tgt_embed)

        encoder_output = self.encoder(src_embed, pad_mask=src_mask)
        decoder_output = self.decoder(
            tgt_embed, encoder_output, src_mask=src_mask, tgt_mask=tgt_mask
        )

        output = self.fc(decoder_output)
        return output
