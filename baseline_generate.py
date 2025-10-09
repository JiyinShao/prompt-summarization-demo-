import os
import json
from llm_utils import summarize_with_prompt
from evaluation import evaluate_summary

RESULT_DIR = "results"
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(RESULT_DIR, "baseline.json")

def load_data():
    with open(os.path.join(DATA_DIR, "cnn_input.json"), "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open(os.path.join(DATA_DIR, "xsum_input.json"), "r", encoding="utf-8") as f:
        xsum_data = json.load(f)
    datasets = {
        "cnn": [(item["article"], item["reference"]) for item in cnn_data],
        "xsum": [(item["article"], item["reference"]) for item in xsum_data]
    }
    return datasets

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    datasets = load_data()
    results = []

    for dataset_name, pairs in datasets.items():
        print(f"Running baseline summarization for {dataset_name} ({len(pairs)} articles)")
        for idx, (article, ref) in enumerate(pairs):
            baseline_prompt = "" 
            summary = summarize_with_prompt(article, baseline_prompt)     
            scores = evaluate_summary(summary, ref, article)
            compression = len(summary.split()) / max(1, len(article.split()))
            record = {
                "dataset": dataset_name,
                "index": idx,
                "summary": summary,
                "rouge1": scores["rouge1"],
                "rougel": scores["rougel"],
                "fre": scores["fre"],
                "compression": compression
            }
            results.append(record)
            print(f"[{dataset_name}] {idx+1}/{len(pairs)} | R1={record['rouge1']:.3f} RL={record['rougel']:.3f}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nBaseline results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
