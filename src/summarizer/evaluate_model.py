import os
import json
import yaml
import torch
import evaluate
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ModelEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.model_dir = config["training"]["final_model_dir"]
        self.processed_dir = config["data"]["processed_dir"]
        self.eval_cfg = config["evaluation"]
        self.max_input_length = config["model"]["max_input_length"]
        self.max_target_length = config["model"]["max_target_length"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()
        self.rouge = evaluate.load("rouge")

    def generate_batch_summaries(self, texts: list[str]) -> list[str]:
        inputs = self.tokenizer(
            texts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_target_length,
                num_beams=self.eval_cfg["num_beams"],
                length_penalty=self.eval_cfg["length_penalty"],
                early_stopping=True,
            )

        return self.tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def run(self, batch_size: int = 8):
        print(f"Loading test dataset on {self.device}...")
        test_ds = load_from_disk(os.path.join(self.processed_dir, "test_raw"))
        dialogues = test_ds["dialogue"]
        references = test_ds["summary"]

        predictions = []
        for i in tqdm(range(0, len(dialogues), batch_size), desc="Evaluating ROUGE"):
            batch_texts = dialogues[i : i + batch_size]
            predictions.extend(self.generate_batch_summaries(batch_texts))

        # Compute ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum
        scores = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )
        rounded_scores = {k: round(v * 100, 2) for k, v in scores.items()}

        # Save qualitative examples for output error analysis
        sample_analysis = []
        for idx in range(min(10, len(dialogues))):
            sample_analysis.append(
                {
                    "sample_id": idx + 1,
                    "dialogue": dialogues[idx],
                    "reference_summary": references[idx],
                    "generated_summary": predictions[idx],
                }
            )

        report = {
            "model_path": self.model_dir,
            "num_test_samples": len(dialogues),
            "rouge_scores": rounded_scores,
            "qualitative_samples": sample_analysis,
        }

        results_path = self.eval_cfg["results_file"]
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print("\n=== ROUGE Evaluation Results ===")
        for metric, value in rounded_scores.items():
            print(f"{metric.upper()}: {value}")
        print(f"\nFull evaluation & qualitative sample analysis saved to '{results_path}'.")


if __name__ == "__main__":
    cfg = load_config()
    evaluator = ModelEvaluator(cfg)
    evaluator.run()