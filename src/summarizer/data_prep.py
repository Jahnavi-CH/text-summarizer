import os
import re
import yaml
import nltk
from datasets import load_dataset
from transformers import AutoTokenizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text_with_nltk(text: str) -> str:
    """Normalizes whitespace and cleans sentence boundaries using NLTK."""
    if not text:
        return ""

    nltk_data_dir = "/tmp/nltk_data"
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data_dir)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
        nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)

    # Normalize irregular whitespace and newlines
    text = re.sub(r"\s+", " ", text).strip()
    sentences = nltk.sent_tokenize(text)
    return " ".join(sentences)


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.max_input_length = config["model"]["max_input_length"]
        self.max_target_length = config["model"]["max_target_length"]

        # Ensure NLTK punkt tokenizer data is available
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def preprocess_batch(self, batch: dict) -> dict:
        # 1. Clean raw dialogues and target summaries with NLTK
        cleaned_dialogues = [clean_text_with_nltk(doc) for doc in batch["dialogue"]]
        cleaned_summaries = [clean_text_with_nltk(summ) for summ in batch["summary"]]

        # 2. Tokenize inputs with explicit padding and truncation
        model_inputs = self.tokenizer(
            cleaned_dialogues,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
        )

        # 3. Tokenize target summaries
        labels = self.tokenizer(
            text_target=cleaned_summaries,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
        )

        # Replace pad_token_id with -100 so PyTorch CrossEntropyLoss ignores padding tokens
        label_ids = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids
        return model_inputs

    def run(self):
        data_cfg = self.config["data"]
        print(f"Loading dataset: {data_cfg['dataset_name']}...")
        dataset = load_dataset(data_cfg["dataset_name"])

        # Filter out any null rows and select reproducible subsets
        train_ds = dataset["train"].filter(lambda x: x["dialogue"] and x["summary"])
        val_ds = dataset["validation"].filter(lambda x: x["dialogue"] and x["summary"])
        test_ds = dataset["test"].filter(lambda x: x["dialogue"] and x["summary"])

        train_ds = train_ds.select(range(min(data_cfg["train_subset_size"], len(train_ds))))
        val_ds = val_ds.select(range(min(data_cfg["val_subset_size"], len(val_ds))))
        test_ds = test_ds.select(range(min(data_cfg["test_subset_size"], len(test_ds))))

        print("Cleaning text with NLTK and tokenizing sequences (padding + truncation)...")
        tokenized_train = train_ds.map(self.preprocess_batch, batched=True, remove_columns=train_ds.column_names)
        tokenized_val = val_ds.map(self.preprocess_batch, batched=True, remove_columns=val_ds.column_names)

        # Save tokenized train/val splits and cleaned raw test split for ROUGE evaluation
        processed_dir = data_cfg["processed_dir"]
        os.makedirs(processed_dir, exist_ok=True)
        tokenized_train.save_to_disk(os.path.join(processed_dir, "train"))
        tokenized_val.save_to_disk(os.path.join(processed_dir, "validation"))
        test_ds.save_to_disk(os.path.join(processed_dir, "test_raw"))

        print(f"Saved processed datasets to '{processed_dir}'.")


if __name__ == "__main__":
    cfg = load_config()
    preprocessor = DataPreprocessor(cfg)
    preprocessor.run()