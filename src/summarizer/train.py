import os
import yaml
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model"]["name"]
        self.train_cfg = config["training"]
        self.processed_dir = config["data"]["processed_dir"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def run(self):
        print("Loading tokenized train and validation datasets...")
        train_ds = load_from_disk(os.path.join(self.processed_dir, "train"))
        val_ds = load_from_disk(os.path.join(self.processed_dir, "validation"))

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
        )

        use_fp16 = torch.cuda.is_available()
        print(f"Hardware detected: {'CUDA GPU (fp16 enabled)' if use_fp16 else 'CPU'}")

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.train_cfg["output_dir"],
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=float(self.train_cfg["learning_rate"]),
            per_device_train_batch_size=self.train_cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=self.train_cfg["per_device_eval_batch_size"],
            weight_decay=self.train_cfg["weight_decay"],
            warmup_steps=self.train_cfg["warmup_steps"],
            num_train_epochs=self.train_cfg["num_train_epochs"],
            predict_with_generate=True,
            fp16=use_fp16,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )

        print("Starting model fine-tuning...")
        trainer.train()

        # Save the best fine-tuned model and tokenizer
        final_dir = self.train_cfg["final_model_dir"]
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        print(f"Fine-tuned model and tokenizer saved to '{final_dir}'.")


if __name__ == "__main__":
    cfg = load_config()
    model_trainer = ModelTrainer(cfg)
    model_trainer.run()