# Transformer

A PyTorch implementation of the Transformer architecture for English-German machine translation using the Multi30k dataset.

## Features

- **Pre-LayerNorm Transformer**: Implements Pre-LayerNorm (normalization before attention/FFN) for improved training stability
- **Multi-Head Self-Attention**: Parallel attention mechanisms with multiple representation subspaces
- **Encoder-Decoder Architecture**: Full sequence-to-sequence model with masking
- **Mixed Precision Training**: Optional AMP support via `torch.cuda.amp` for faster training
- **Xavier Uniform Initialization**: Proper weight initialization for stable training
- **Gradient Monitoring**: Built-in NaN/Inf detection to catch training instability

## Installation

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

## Quick Start

### Training

Train the model with default parameters:

```bash
uv run scripts/train.py
```

Train with custom parameters:

```bash
uv run scripts/train.py --batch_size 16 --use_amp --epochs 50 --warmup_steps 10000
```

**Important**: High `warmup_steps` (e.g., 10000) is critical for training stability with this architecture.

### Inference

Run inference on trained model:

```bash
uv run scripts/inference.py
```

### Testing

Run the transformer smoke test:

```bash
uv run tests/test_transformer.py
```

## Architecture

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query), $K$ (Key), $V$ (Value) are input matrices
- $d_k$ is the dimension of the key vectors
- Scaling by $\sqrt{d_k}$ prevents softmax saturation

### Multi-Head Attention

Multiple attention heads run in parallel, allowing the model to attend to different representation subspaces:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### Encoder

Stack of N encoder blocks, each containing:
1. Multi-Head Self-Attention
2. Position-wise Feed-Forward Network
3. Residual connections and LayerNorm (Pre-LayerNorm)

### Decoder

Stack of N decoder blocks, each containing:
1. Masked Multi-Head Self-Attention
2. Encoder-Decoder Cross-Attention
3. Position-wise Feed-Forward Network
4. Residual connections and LayerNorm (Pre-LayerNorm)

## Project Structure

```
transformer/
├── src/transformer/
│   ├── model/          # Model components
│   │   ├── transformer.py    # Main Transformer class
│   │   ├── encoder.py        # Encoder blocks
│   │   ├── decoder.py        # Decoder blocks
│   │   └── attention.py      # Attention mechanisms
│   ├── data/           # Dataset handling
│   └── engines/        # Training and validation loops
├── scripts/            # Entry points
│   ├── train.py       # Training script
│   └── inference.py   # Inference script
├── tests/             # Manual verification scripts
└── checkpoint_dir/    # Model checkpoints
```

## Key Implementation Details

- **Padding**: The model correctly handles padding tokens via `padding_idx` in embeddings
- **Teacher Forcing**: During training, target input is `tgt[:, :-1]` and labels are `tgt[:, 1:]`
- **Vocabulary Size**: Uses `len(tokenizer)` instead of `tokenizer.vocab_size` to account for special tokens
- **Tokenizer**: Uses MarianTokenizer from Hugging Face transformers
- **Dataset**: Multi30k (English-German translation pairs)

## Training Tips

- Start with high warmup steps (10000+) to prevent gradient explosion
- Use mixed precision (`--use_amp`) for faster training on modern GPUs
- Monitor for NaN/Inf losses - the training loop will catch and report these
- If training diverges, increase warmup steps or reduce learning rate

## Requirements

- Python >= 3.11
- PyTorch >= 2.10.0
- CUDA-capable GPU (recommended for training)

## License

See LICENSE file for details.
