from prompt_generator import generate_prompt
from llm_interface import query_llm
from evaluator import evaluate_fre, evaluate_rouge
from datasets import load_dataset
from mutations import MUTATIONS
from config import (
    DATASETS,
    PROMPT_TEMPLATES,
    SAMPLES_PER_DATASET,
    SEED,
    RESULT_DIR,
)
import json, os, time, random

random.seed(SEED)
os.makedirs(RESULT_DIR, exist_ok=True)

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

def all_candidates(active_mutations: list[str]):
    return [(t, m) for t in PROMPT_TEMPLATES for m in active_mutations]

def apply_mutation_chain(m_spec: str, prompt: str) -> str:
    names = m_spec.split("+") if m_spec else ["none"]
    out = prompt
    for name in names:
        fn = MUTATIONS.get(name, lambda x: x)
        out = fn(out)
    return out

def run_dataset(
    dataset_name: str,
    k: int,
    candidates: list[tuple[str, str]] | None = None,
    active_mutations: list[str] | None = None,
    samples: list[dict] | None = None,  
):
    if samples is None:
        samples = load_samples(dataset_name, k)
    else:
        samples = list(samples)

    pairs = candidates if candidates is not None else all_candidates(active_mutations or list(MUTATIONS.keys()))
    results = []
    t0 = time.time()
    total = len(samples) * len(pairs)
    step = 0

    for template_name, m_name in pairs:
        for sample in samples:
            step += 1
            try:
                base_prompt = generate_prompt(sample["article"], template_file=template_name)
                mutated_prompt = apply_mutation_chain(m_name, base_prompt)
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
                    "rouge1": rouge.get("rouge1"),
                    "rougeL": rouge.get("rougeL"),
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

    out_path = os.path.join(RESULT_DIR, f"{dataset_name}_prompt_eval_{len(samples)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {out_path}")
    return results

def run_once(
    datasets: list[str] = DATASETS,
    samples_per_dataset: int = SAMPLES_PER_DATASET,
    candidates: list[tuple[str, str]] | None = None,
    active_mutations: list[str] | None = None,
    samples_map: dict[str, list[dict]] | None = None, 
):
    all_results = []
    for ds in datasets:
        fixed_samples = None if samples_map is None else samples_map.get(ds)
        res = run_dataset(
            ds,
            samples_per_dataset,
            candidates=candidates,
            active_mutations=active_mutations,
            samples=fixed_samples, 
        )
        all_results.extend(res)
    return all_results

if __name__ == "__main__":
    run_once()
