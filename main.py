from prompt_generator import generate_prompt
from llm_interface import query_llm
from evaluator import evaluate_fre, evaluate_rouge
from datasets import load_dataset
from mutations import MUTATIONS
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
ACTIVE_MUTATIONS = [
    "none",
    "synonym_replacement",
    "prompt_rewriting",
    "style_instruction",
    "audience_information",
    "stepwise_prompt",
]

SEED = 42
random.seed(SEED)

os.makedirs("results", exist_ok=True)

def load_samples(name: str, k: int):
    if name == "cnn":
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
        items = random.sample(list(ds), min(k, len(ds)))
        return [
            {"id": f"{name}_{i+1:03}", "article": s["article"], "reference": s["highlights"]}
            for i, s in enumerate(items)
        ]
    elif name == "xsum":
        try:
            ds = load_dataset("EdinburghNLP/xsum", split="test", revision="refs/convert/parquet")
        except Exception:
            ds = load_dataset("EdinburghNLP/xsum", split="test", trust_remote_code=True)
        items = random.sample(list(ds), min(k, len(ds)))
        return [
            {"id": f"{name}_{i+1:03}", "article": s["document"], "reference": s["summary"]}
            for i, s in enumerate(items)
        ]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def run_dataset(dataset_name: str, k: int):
    samples = load_samples(dataset_name, k)
    results = []
    t0 = time.time()
    total = len(samples) * len(PROMPT_TEMPLATES) * len(ACTIVE_MUTATIONS)
    step = 0

    for template_name in PROMPT_TEMPLATES:
        for m_name in ACTIVE_MUTATIONS:
            m_fn = MUTATIONS[m_name]
            for sample in samples:
                step += 1
                try:
                    base_prompt = generate_prompt(sample["article"], template_file=template_name)
                    mutated_prompt = m_fn(base_prompt)
                    output = query_llm(mutated_prompt).lstrip(": ").strip()

                    fre = evaluate_fre(output)
                    rouge = evaluate_rouge(output, sample["reference"])
                    comp_ratio = (len(output) + 1e-9) / (len(sample["article"]) + 1e-9)

                    results.append({
                        "id": sample["id"],
                        "dataset": dataset_name,
                        "template": template_name,
                        "mutation": m_name,
                        "prompt": mutated_prompt,
                        "output": output,
                        "fre": fre,
                        "rouge1": rouge["rouge1"],
                        "rougeL": rouge["rougeL"],
                        "compression_ratio": comp_ratio,
                        "seed": SEED,
                    })
                except Exception as e:
                    results.append({
                        "id": sample["id"],
                        "dataset": dataset_name,
                        "template": template_name,
                        "mutation": m_name,
                        "error": str(e),
                        "seed": SEED,
                    })

                if step % 10 == 0 or step == total:
                    print(f"[{dataset_name}] {step}/{total} | elapsed: {time.time() - t0:.1f}s")

    out_path = f"results/{dataset_name}_prompt_eval_{len(samples)}_with_mutations.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    for ds in DATASETS:
        run_dataset(ds, SAMPLES_PER_DATASET)
