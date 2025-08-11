from prompt_generator import generate_prompt
from llm_interface import query_llm
from evaluator import evaluate_fre, evaluate_rouge
from datasets import load_dataset
import json, os, time, random

DATASETS = ["cnn", "xsum"]
PROMPT_TEMPLATES = [
    "zero_shot.txt",
    "few_shot.txt",
    "instruction_based.txt",
    "pattern_based.txt",
    "target_audience.txt",
]
SAMPLES_PER_DATASET = 5

os.makedirs("results", exist_ok=True)

def load_samples(name: str, k: int):
    if name == "cnn":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        items = random.sample(list(ds), min(k, len(ds)))
        return [{"id": f"{name}_{i+1:03}", "article": s["article"], "reference": s["highlights"]} for i, s in enumerate(items)]
    elif name == "xsum":
        try:
            ds = load_dataset("EdinburghNLP/xsum", split="test", revision="refs/convert/parquet")
        except Exception:
            ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
        items = random.sample(list(ds), min(k, len(ds)))
        return [{"id": f"{name}_{i+1:03}", "article": s["document"], "reference": s["summary"]} for i, s in enumerate(items)]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def run_dataset(dataset_name: str, k: int):
    samples = load_samples(dataset_name, k)
    results = []
    t0 = time.time()
    total = len(samples) * len(PROMPT_TEMPLATES)
    step = 0

    for template_name in PROMPT_TEMPLATES:
        for sample in samples:
            step += 1
            prompt = generate_prompt(sample["article"], template_file=template_name)
            output = query_llm(prompt).lstrip(": ").strip()
            fre = evaluate_fre(output)
            rouge = evaluate_rouge(output, sample["reference"])
            results.append({
                "id": sample["id"],
                "dataset": dataset_name,
                "template": template_name,
                "output": output,
                "fre": fre,
                "rouge1": rouge["rouge1"],
                "rougeL": rouge["rougeL"],
            })
            if step % 5 == 0 or step == total:
                print(f"[{dataset_name}] {step}/{total} done | elapsed: {time.time() - t0:.1f}s")

    out_path = f"results/{dataset_name}_prompt_eval_{len(samples)}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    for ds in DATASETS:
        run_dataset(ds, SAMPLES_PER_DATASET)
