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
    vocab_size = len(train_dataset.tokenizer)  # Use len() to match training
    pad_token_id = train_dataset.tokenizer.pad_token_id
    max_len = train_dataset.max_len
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch = next(iter(train_loader))
    print(batch["source_ids"].shape)
    print(batch["target_ids"].shape)
    print(batch["source_mask"].shape)
    print(batch["source_mask"][0])

    src = batch["source_ids"].to(device)
    tgt = batch["target_ids"].to(device)

    # Match training mask generation logic
    src_mask = batch["source_mask"].unsqueeze(1).unsqueeze(1).to(device)
    
    # For target: use teacher forcing setup (input is tgt[:-1])
    tgt_input = tgt[:, :-1]
    tgt_len = tgt_input.size(1)
    tgt_pad_mask = (tgt_input != pad_token_id).unsqueeze(1).unsqueeze(1)
    tgt_causal = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
    tgt_mask = tgt_pad_mask & tgt_causal.unsqueeze(0).unsqueeze(0)

    model = Transformer(
        vocab_size=vocab_size, 
        embed_dim=512, 
        mid_dim=2048,
        padding_idx=pad_token_id
    )
    model = model.to(device)
    model.eval()

    summary(model, input_data=(src, tgt_input, src_mask, tgt_mask))
