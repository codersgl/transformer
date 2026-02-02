import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_k: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        """
        Scale Dot-Product Attention

        Args:
            Q: [batch_size, num_heads, seq_len_q, d_k]
            K: [batch_size, num_heads, seq_len_k, d_k]
            V: [batch_size, num_heads, seq_len_v, d_k]
            mask: 可选，形状为 [batch_size, num_heads, seq_len_q, seq_len_k]
                  或可广播的形状（如 [batch_size, 1, seq_len_q, seq_len_k]）
        """
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(
            self.d_k
        )  # [batch_size, num_heads, seq_len_q, seq_len_k]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)

        attention_weights = F.softmax(score, dim=-1)

        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights)

        attention_output = torch.matmul(
            attention_weights, V
        )  # [batch_size, num_heads, seq_len_q, d_k]

        return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.attention = ScaleDotProductAttention(self.d_k, dropout)

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)
        self.W_O = nn.Linear(embed_dim, embed_dim)

    def _prepare_mask_for_attention(
        self, mask: torch.Tensor | None, batch_size: int, seq_len_q: int, seq_len_k: int
    ) -> torch.Tensor | None:
        if mask is None:
            return None

        # 确保mask至少是3D [B, S_q, S_k]
        if mask.dim() == 2:
            # [S_q, S_k] -> [1, S_q, S_k] -> [1, 1, S_q, S_k]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # [B, S_q, S_k] -> [B, 1, S_q, S_k]
            mask = mask.unsqueeze(1)

        # 确保mask有正确的batch_size（广播或重复）
        if mask.size(0) == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)

        # 确保mask有正确的head维度（广播到所有头）
        if mask.size(1) == 1 and self.num_heads > 1:
            mask = mask.expand(-1, self.num_heads, -1, -1)

        return mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ):
        """
        Args:
            query: [batch_size, seq_len_q, embed_dim]
            key: [batch_size, seq_len_k, embed_dim] (可选，默认为query)
            value: [batch_size, seq_len_v, embed_dim] (可选，默认为key)
            mask: 可选，形状可以是：
                  - 2D: [seq_len_q, seq_len_k]
                  - 3D: [batch_size, seq_len_q, seq_len_k]
                  - 4D: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        _, seq_len_v, _ = value.size()

        Q_proj = self.W_Q(query)  # [batch_size, seq_len_q, embed_dim]
        K_proj = self.W_K(key)  # [batch_size, seq_len_k, embed_dim]
        V_proj = self.W_V(value)  # [batch_size, seq_len_v, embed_dim]

        Q_heads = Q_proj.view(
            batch_size, seq_len_q, self.num_heads, self.d_k
        )  # [batch_size, seq_len_q, num_heads, d_k]
        K_heads = K_proj.view(
            batch_size, seq_len_k, self.num_heads, self.d_k
        )  # [batch_size, seq_len_k, num_heads, d_k]
        V_heads = V_proj.view(
            batch_size, seq_len_v, self.num_heads, self.d_k
        )  # [batch_size, seq_len_v, num_heads, d_k]

        Q_heads = Q_heads.transpose(
            1, 2
        ).contiguous()  # [batch_size, num_heads, seq_len_q, d_k]
        K_heads = K_heads.transpose(
            1, 2
        ).contiguous()  # [batch_size, num_heads, seq_len_k, d_k]
        V_heads = V_heads.transpose(
            1, 2
        ).contiguous()  # [batch_size, num_heads, seq_len_v, d_k]

        attention_mask = self._prepare_mask_for_attention(
            mask, batch_size, seq_len_q, seq_len_k
        )

        multi_head_output, attention_weights = self.attention(
            Q_heads, K_heads, V_heads, attention_mask
        )  # [batch_size, num_heads, seq_len_q, d_k]

        multi_head_output = multi_head_output.transpose(
            1, 2
        ).contiguous()  # [batch_size, seq_len_q, num_heads, d_k]

        multi_head_output = multi_head_output.view(
            batch_size, seq_len_q, self.embed_dim
        )  # [batch_size, seq_len_q, embed_dim]

        output = self.W_O(multi_head_output)  # [batch_size, seq_len_q, embed_dim]

        if self.dropout is not None:
            output = self.dropout(output)

        if return_attention_weights:
            return output, attention_weights

        return output


if __name__ == "__main__":
    mha = MultiHeadAttention(embed_dim=512, num_heads=8, dropout=0.1)

    x = torch.randn(2, 10, 512)
    output1 = mha(x)  # 自注意力
    print(f"自注意力输出形状: {output1.shape}")

    query = torch.randn(2, 5, 512)
    key = torch.randn(2, 10, 512)
    value = torch.randn(2, 10, 512)
    output2 = mha(query, key, value)
    print(f"交叉注意力输出形状: {output2.shape}")

    # 因果掩码
    causal_mask = torch.tril(torch.ones(10, 10))
    output3 = mha(x, mask=causal_mask)
    print(f"带2D掩码输出形状: {output3.shape}")

    # 批量掩码
    batch_mask = torch.ones(2, 10, 10)
    batch_mask[0, :, 5:] = 0  # 第一个样本后5个位置被mask
    output4 = mha(x, mask=batch_mask)
    print(f"带3D掩码输出形状: {output4.shape}")

    output5, attn_weights = mha(x, return_attention_weights=True)
    print(f"注意力权重形状: {attn_weights.shape}")  # [2, 8, 10, 10]
