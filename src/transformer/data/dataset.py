from typing import Dict

import torch
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from transformers import MarianTokenizer


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str,
        model_name: str = "Helsinki-NLP/opus-mt-en-de",
        max_len: int = 128,
        cache_dir: str = "./cache",
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.cache_dir = cache_dir

        logger.info(f"Loading Multi30k {split} split...")
        self.dataset = load_dataset("bentrevett/multi30k", split=split)

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)

        self.processed_data = self._preprocess_dataset()

        print(f"Dataset ready: {len(self)} samples")

    def _preprocess_dataset(self):
        def tokenize_batch(examples):
            encoded = self.tokenizer(
                examples["en"],
                text_target=examples["de"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )

            return {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encoded["labels"],
            }

        processed = self.dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["en", "de"],
            desc="Tokenizing with MarianTokenizer",
        )

        processed.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"],
        )

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.processed_data[idx]

        return {
            "source_ids": item["input_ids"],
            "source_mask": item["attention_mask"],
            "target_ids": item["labels"],
        }

    def get_sample_text(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    train_dataset = Multi30kDataset("train[:5000]")
    val_dataset = Multi30kDataset("validation")

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    vocab_size = train_dataset.tokenizer.vocab_size
    print(f"vocab_size: {vocab_size}")

    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Source IDs shape: {batch['source_ids'].shape}")
    print(f"Target IDs shape: {batch['target_ids'].shape}")

    print("\nSample decoded:")
    source_text = train_dataset.tokenizer.decode(batch["source_ids"][0])
    target_text = train_dataset.tokenizer.decode(batch["target_ids"][0])
    print(f"Source: {source_text[:100]}...")
    print(f"Target: {target_text[:100]}...")
