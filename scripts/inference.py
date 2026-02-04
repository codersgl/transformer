import argparse
import random

import numpy as np
import torch
from transformers import MarianTokenizer

from transformer.model.transformer import Transformer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Inference stript for translate")
    parser.add_argument(
        "--input_sentence", required=True, help="A english sentence for translating"
    )
    parser.add_argument(
        "--model_path", type=str, default="checkpoint_dir/best_model.pth"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Helsinki-NLP/opus-mt-en-de",
        help="Tokenizer Model",
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_encoder_blocks", type=int, default=6)
    parser.add_argument("--num_decoder_blocks", type=int, default=6)
    parser.add_argument(
        "--mid_dim", type=int, default=2048, help="The hidden dims in FeedForwordNet"
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)  # Use len() to match training

    encoded = tokenizer(
        args.input_sentence,
        truncation=True,
        padding="max_length",
        max_length=args.max_len,
        return_tensors="pt",
    )
    src_ids = encoded["input_ids"].to(device)
    src_mask = encoded["attention_mask"].unsqueeze(1).unsqueeze(1).to(device)

    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        mid_dim=args.mid_dim,
        num_heads=args.num_heads,
        num_encoder_blocks=args.num_encoder_blocks,
        num_decoder_blocks=args.num_decoder_blocks,
        dropout=args.dropout,
        padding_idx=pad_token_id,
    ).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    # Start with decoder start token (pad_token_id is used as BOS in MarianTokenizer)
    # Note: MarianTokenizer uses pad_token_id as the decoder start token by convention
    tgt_ids = torch.full((1, 1), pad_token_id, dtype=torch.long, device=device)

    for _ in range(args.max_len):
        tgt_len = tgt_ids.size(1)
        tgt_pad_mask = (tgt_ids != pad_token_id).unsqueeze(1).unsqueeze(1)
        tgt_causal = (
            torch.tril(torch.ones((tgt_len, tgt_len), device=device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        tgt_mask = tgt_pad_mask & tgt_causal

        output = model(src_ids, tgt_ids, src_mask, tgt_mask)
        next_token_logits = output[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

        tgt_ids = torch.cat([tgt_ids, next_token_id], dim=1)

        if (next_token_id == eos_token_id).all():
            break

    print(f"Generated IDs: {tgt_ids}")
    translation = tokenizer.decode(tgt_ids[0], skip_special_tokens=True)
    print(f"\nSource: {args.input_sentence}")
    print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
