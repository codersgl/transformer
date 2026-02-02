import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from transformer.data.dataset import Multi30kDataset
from transformer.model.transformer import Transformer


if __name__ == "__main__":
    train_dataset = Multi30kDataset("train[:5000]")
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    vocab_size = train_dataset.tokenizer.vocab_size  # 58101
    max_len = train_dataset.max_len
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = next(iter(train_loader))
    print(batch["source_ids"].shape)
    print(batch["target_ids"].shape)
    print(batch["source_mask"].shape)
    print(batch["source_mask"][0])

    src = batch["source_ids"].to(device)
    tgt = batch["target_ids"].to(device)

    src_mask = batch["source_mask"].unsqueeze(1).unsqueeze(1)
    tgt_mask = src_mask & torch.tril(
        torch.ones(1, 1, max_len, max_len, dtype=torch.long)
    )

    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    model = Transformer(vocab_size=vocab_size, embed_dim=512, mid_dim=2048)
    model = model.to(device)
    model.eval()

    summary(model, input_data=(src, tgt, src_mask, tgt_mask))
