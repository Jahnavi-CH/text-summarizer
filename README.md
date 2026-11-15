https://jahnavi-text-summarizer.streamlit.app/


# End-to-End Abstractive Text Summarization (BART + PyTorch + Hugging Face)

An end-to-end NLP text summarization pipeline built with Python, PyTorch, Hugging Face Transformers, and NLTK. The system preprocesses conversational text, fine-tunes a sequence-to-sequence `facebook/bart-base` model on the SAMSum dataset, evaluates summary quality using ROUGE metrics, and exposes a reusable inference pipeline for unseen text.

## Project Architecture

```text
text-summarizer/
├── config/
│   └── config.yaml              # Centralized model, data, training, and evaluation configs
├── src/
│   └── summarizer/
│       ├── data_prep.py         # NLTK sentence normalization, tokenization, padding & truncation
│       ├── train.py             # Seq2SeqTrainer fine-tuning & validation workflow (fp16 GPU support)
│       ├── evaluate_model.py    # ROUGE-1/2/L evaluation & qualitative sample output logging
│       └── inference.py         # Reusable SummarizerPipeline class for unseen text
├── artifacts/                   # Saved datasets, fine-tuned weights, and evaluation reports (gitignored)
├── main.py                      # Single CLI entrypoint for prep, train, eval, and infer stages
└── requirements.txt             # Reproducible Python dependencies

### 1. Setup Environment

python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

### 2. Run Pipeline Stages

# 1. Clean text with NLTK and tokenize dataset splits
python main.py prep

# 2. Fine-tune facebook/bart-base (recommended on CUDA GPU)
python main.py train

# 3. Evaluate on unseen test set and generate ROUGE report
python main.py eval

# 4. Run inference on sample unseen text
python main.py infer

## Evaluation Results (SAMSum Test Split — 200 Samples)
Fine-tuned facebook/bart-base for 2 epochs (max_input_length=512, max_target_length=128, num_beams=4, length_penalty=1.0)

Metric,Score
ROUGE-1,41.76
ROUGE-2,19.57
ROUGE-L,35.64
ROUGE-Lsum,35.63

## Error Analysis & Future Improvements
Qualitative Finding: While the model captures key entities and main intent fluently, multi-turn dialogues with 3+ speakers occasionally exhibit speaker-attribution confusion (merging actions across speakers).

Next Steps: Scale training to the full 14.7k SAMSum training split, experiment with facebook/bart-large-cnn, and wrap SummarizerPipeline in a FastAPI endpoint for real-time serving.
