import yaml
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.summarizer.data_prep import clean_text_with_nltk


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SummarizerPipeline:
    """Reusable inference pipeline for generating summaries from unseen text."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.model_dir = self.config["training"]["final_model_dir"]
        self.max_input_length = self.config["model"]["max_input_length"]
        self.max_target_length = self.config["model"]["max_target_length"]
        self.eval_cfg = self.config["evaluation"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

    def summarize(self, text: str, max_length: int | None = None) -> str:
        cleaned_text = clean_text_with_nltk(text)
        if not cleaned_text:
            return ""

        inputs = self.tokenizer(
            cleaned_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length or self.max_target_length,
                num_beams=self.eval_cfg["num_beams"],
                length_penalty=self.eval_cfg["length_penalty"],
                early_stopping=True,
            )

        return self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


if __name__ == "__main__":
    sample_dialogue = """
    Sarah: Hey Alex, are we still on for the client demo tomorrow at 3 PM?
    Alex: Actually, the client asked to move it to 4:30 PM because their VP had a conflict.
    Sarah: Got it. Does that give us enough time to test the new API endpoint before the call?
    Alex: Yes, Priya and I are running final load tests at 2 PM, so everything will be ready by 3:30.
    Sarah: Perfect, I'll update the calendar invite and send over the revised slide deck.
    """

    summarizer = SummarizerPipeline()
    summary = summarizer.summarize(sample_dialogue)
    print("\n--- Input Text ---")
    print(sample_dialogue.strip())
    print("\n--- Generated Summary ---")
    print(summary)