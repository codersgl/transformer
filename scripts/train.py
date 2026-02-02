import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformer.data.dataset import Multi30kDataset
from transformer.engines.train import train_one_epoch, valid_one_epoch
from transformer.model.misc import NoamScheduler
from transformer.model.transformer import Transformer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="en-de translator based on Transformer",
        epilog="train.py --batch_size 8 --use_amp",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=6)
    parser.add_argument("--num_decoder_blocks", type=int, default=6)
    parser.add_argument("--log_dir", type=str, default="runs/transformer")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoint_dir")
    parser.add_argument("--use_amp", action="store_true", help="使用混合精度训练")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-de",
        help="Tokenizer Model",
    )
    parser.add_argument(
        "--mid_dim", type=int, default=2048, help="The hidden dims in FeedForwordNet"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)

    args = parser.parse_args()

    set_seed(args.seed)

    train_dataset = Multi30kDataset(
        "train", model_name=args.model_name, max_len=args.max_len
    )

    valid_dataset = Multi30kDataset(
        "validation", model_name=args.model_name, max_len=args.max_len
    )

    vocab_size = train_dataset.tokenizer.vocab_size
    padding_index = train_dataset.tokenizer.pad_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logger.info(f"ArgumentParser: {args}")
    logger.info(f"Using device: {device}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        mid_dim=args.mid_dim,
        num_heads=args.num_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        dropout=args.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=padding_index, label_smoothing=args.label_smoothing
    )
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scheduler = NoamScheduler(optimizer, args.embed_dim, args.warmup_steps)

    scaler = GradScaler() if args.use_amp and torch.cuda.is_available() else None
    writer = SummaryWriter(log_dir=args.log_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    patience = 3
    counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            epoch=epoch + 1,
            writer=writer,
            scheduler=scheduler,
            scaler=scaler,
            vocab_size=vocab_size,
            padding_index=padding_index,
            device=device,
            use_amp=args.use_amp,
        )
        valid_loss = valid_one_epoch(
            model,
            valid_dataloader,
            loss_fn,
            vocab_size,
            padding_index,
            device=device,
            use_amp=args.use_amp,
        )

        writer.add_scalar("Epoch/Loss", train_loss, epoch)
        writer.add_scalar("Epoch/LR", scheduler._get_lr(), epoch)
        writer.add_scalar("Valid/Loss", valid_loss, epoch)

        if valid_loss < best_loss - 1e-4:
            best_loss = valid_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pth")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stop！Best loss: {best_loss:.4f}")
                break

        if (epoch + 1) % 5 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                },
                checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth",
            )

        logger.info(
            f"Epoch {epoch + 1}, train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}, lr: {scheduler._get_lr():.4f}"
        )


if __name__ == "__main__":
    main()
