import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    scheduler,
    device: torch.device,
    vocab_size: int,
    padding_index: int,
    scaler,
    writer,
    epoch: int,
    use_amp: bool = False,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        src_ids = batch["source_ids"].to(device, non_blocking=True)
        tgt_ids = batch["target_ids"].to(device, non_blocking=True)
        src_mask = (
            batch["source_mask"].unsqueeze(1).unsqueeze(1).to(device, non_blocking=True)
        )

        tgt_len = tgt_ids.size(1) - 1

        # Teacher Forcing
        tgt_input = tgt_ids[:, :-1]
        tgt_label = tgt_ids[:, 1:]

        # target mask
        tgt_pad_mask = (tgt_input != padding_index).unsqueeze(1).unsqueeze(1)
        tgt_causal = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        tgt_mask = tgt_pad_mask & tgt_causal.unsqueeze(0).unsqueeze(0)

        with autocast(device.type, enabled=use_amp):
            pred = model(src_ids, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(pred.reshape(-1, vocab_size), tgt_label.reshape(-1))

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 先 unscale 才能裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        loss_item = loss.item()
        total_loss += loss_item

        global_step = epoch * len(dataloader) + batch_idx
        if batch_idx % 10 == 0:
            writer.add_scalar("Train/Loss", loss_item, global_step)
            writer.add_scalar("Train/LR", scheduler._get_lr(), global_step)

        pbar.set_postfix(
            {"loss": f"{loss_item:.4f}", "lr": f"{scheduler._get_lr():.6f}"}
        )

    return total_loss / len(dataloader)


@torch.no_grad()
def valid_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,  # CrossEntropyLoss
    vocab_size: int,
    padding_index: int,
    device: torch.device,
    use_amp: bool = False,
):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Valid")
    for batch in pbar:
        src_ids = batch["source_ids"].to(device)  # [batch, src_len]
        tgt_ids = batch["target_ids"].to(device)  # [batch, tgt_len]
        src_mask = (
            batch["source_mask"].unsqueeze(1).unsqueeze(1).to(device)
        )  # [B,1,1,S]

        tgt_len = tgt_ids.size(1) - 1  # 去掉一个

        # Teacher Forcing：输入和标签错开一位
        tgt_input = tgt_ids[:, :-1]  # [batch, tgt_len-1]，去掉最后一个
        tgt_label = tgt_ids[:, 1:]  # [batch, tgt_len-1]，去掉第一个 <bos>

        tgt_pad_mask = ((tgt_input != padding_index).unsqueeze(1).unsqueeze(1)).to(
            device
        )  # [B, 1, 1, 127]
        tgt_mask = tgt_pad_mask & (
            torch.tril(torch.ones(tgt_len, tgt_len, device=src_ids.device))
            .unsqueeze(0)
            .unsqueeze(0)
            .bool()
        ).to(device)  # [1,1,T,T]

        with autocast(device_type=device.type, enabled=use_amp):
            pred = model(src_ids, tgt_input, src_mask, tgt_mask)  # [B, T-1, vocab]
            loss = loss_fn(pred.reshape(-1, vocab_size), tgt_label.reshape(-1))

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)
