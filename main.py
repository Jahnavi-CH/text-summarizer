import sys
from src.summarizer.data_prep import load_config, DataPreprocessor
from src.summarizer.train import ModelTrainer
from src.summarizer.evaluate_model import ModelEvaluator
from src.summarizer.inference import SummarizerPipeline


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "infer"
    cfg = load_config()

    if stage == "prep":
        DataPreprocessor(cfg).run()
    elif stage == "train":
        ModelTrainer(cfg).run()
    elif stage == "eval":
        ModelEvaluator(cfg).run()
    elif stage == "infer":
        summarizer = SummarizerPipeline()
        sample = (
            "Hannah: Hey, do you have Betty's number? "
            "Amanda: Lemme check. Sorry, can't find it. Ask Larry, he called her last time we were at the park together. "
            "Hannah: I don't know him well. Amanda: Don't be shy, he's very nice. Hannah: If you say so.. I'd rather you texted him. "
            "Amanda: Just text him 🙂 Hannah: Urgh.. Alright. Bye!"
        )
        print("Summary:", summarizer.summarize(sample))
    else:
        print("Usage: python main.py [prep | train | eval | infer]")


if __name__ == "__main__":
    main()