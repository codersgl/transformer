from typing import Tuple

import torch
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(
        self, embed_dim: int = 512, mid_dim: int = 2048, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


def get_mask(
    src_mask: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        src_mask: (batch_size, max_len)
    Returns:
        src_mask: (batch_size, 1, 1, max_len)
        tgt_mask: (batch_size, 1, 1, max_len) & (batch_size, 1, max_len, max_len)
    """
    batch_size, max_len = src_mask.size()
    src_mask = src_mask.unsqueeze(1).unsqueeze(1)
    tgt_mask = src_mask & torch.tril(
        torch.ones(batch_size, 1, max_len, max_len, dtype=torch.long)
    )

    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    return src_mask, tgt_mask


class NoamScheduler:
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _get_lr(self):
        # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        d_model = self.d_model
        step = self.step_num
        warmup = self.warmup_steps

        return (d_model**-0.5) * min(
            step**-0.5,  # 衰减部分
            step * (warmup**-1.5),  # warmup 部分
        )

    def state_dict(self):
        return {
            "step_num": self.step_num,
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state_dict):
        self.step_num = state_dict["step_num"]
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
