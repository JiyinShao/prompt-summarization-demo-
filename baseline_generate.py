import json
import os
from llm_utils import MODEL_NAME, summarize_with_prompt
from evaluation import evaluate_summary

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(RESULT_DIR, "baseline.json")

def run_baseline(datasets):
    results = []
    for dataset_name, samples in datasets.items():
        print(f"\n=== Running baseline summarization for {dataset_name} ({len(samples)} articles) ===")
        for idx, (article, ref) in enumerate(samples):
            summary = summarize_with_prompt(article, prompt_text=None)
            scores = evaluate_summary(summary, ref, article)
            record = {
                "dataset": dataset_name,
                "index": idx,
                "summary": summary,
                "rouge1": scores["rouge1"],
                "rougel": scores["rougel"],
                "fre": scores["fre"],
                "compression": scores["compression"]
            }
            results.append(record)
            print(f"[{dataset_name}] {idx+1}/{len(samples)} | R1={record['rouge1']:.3f} RL={record['rougel']:.3f}")
    return results

if __name__ == "__main__":
    print(f"Using model: {MODEL_NAME}")
    with open("data/cnn_input.json", "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open("data/xsum_input.json", "r", encoding="utf-8") as f:
        xsum_data = json.load(f)
    datasets = {
        "cnn": [(item["article"], item["reference"]) for item in cnn_data],
        "xsum": [(item["article"], item["reference"]) for item in xsum_data]
    }

    results = run_baseline(datasets)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nBaseline results saved to {OUTPUT_FILE}")
