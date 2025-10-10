import json
import os
from mutations import mutate
from llm_utils import summarize_with_prompt
from evaluation import evaluate_summary

MUTATION_NAMES = [
    "synonym_replacement",
    "prompt_rewriting",
    "style_instruction",
    "audience_information",
    "stepwise_prompt"
]

def _normalize_population(initial_prompts):
    norm = []
    for i, p in enumerate(initial_prompts):
        if isinstance(p, dict):
            name = p.get("name", f"p{i+1}")
            text = p.get("text", "")
        else:
            name = f"p{i+1}"
            text = str(p)
        norm.append({"name": name, "text": text})
    return norm

def run_evolution(datasets, initial_prompts, max_rounds=5, top_k=5, rouge1_threshold=0.5):
    population = _normalize_population(initial_prompts)
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    meet_file = os.path.join(RESULT_DIR, "meet_threshold.json")
    selected_file = os.path.join(RESULT_DIR, "selected_topk.json")
    open(meet_file, "w").close()
    open(selected_file, "w").close()

    for round_idx in range(1, max_rounds + 1):
        print(f"\n=== Round {round_idx} | candidates={len(population)} ===")
        candidates = []
        for parent in population:
            candidates.append({
                "round": round_idx,
                "parent_name": parent["name"],
                "mutation": "none", 
                "prompt_text": parent["text"]
            })
            for mtype in MUTATION_NAMES:
                mutated_list = mutate(parent["text"], mtype)
                for mp in mutated_list:
                    candidates.append({
                        "round": round_idx,
                        "parent_name": parent["name"],
                        "mutation": mtype,
                        "prompt_text": mp
                    })
        print(f"[round {round_idx}] candidates={len(candidates)}")

        results = []
        for dataset_name, samples in datasets.items():
            for article, ref in samples:
                for cand in candidates:
                    summary = summarize_with_prompt(article, cand["prompt_text"])
                    scores = evaluate_summary(summary, ref, article)
                    results.append({
                        "round": cand["round"],
                        "dataset": dataset_name,
                        "prompt_name": cand["parent_name"],  
                        "mutation": cand["mutation"],
                        "prompt_text": cand["prompt_text"],
                        "summary": summary,
                        "rouge1": scores["rouge1"],
                        "rougel": scores["rougel"],
                        "fre": scores["fre"],
                        "compression": scores["compression"]
                    })
        print(f"[round {round_idx}] results={len(results)}")

        with open(os.path.join(RESULT_DIR, f"round_{round_idx}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        meet = [r for r in results if r["rouge1"] >= rouge1_threshold]
        remain = [r for r in results if r["rouge1"] < rouge1_threshold]

        if meet:
            with open(meet_file, "a", encoding="utf-8") as f:
                for item in meet:
                    item["meet_round"] = round_idx
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        remain.sort(key=lambda x: x["rouge1"], reverse=True)
        selected = remain[:top_k]

        if selected:
            with open(selected_file, "a", encoding="utf-8") as f:
                for item in selected:
                    record = {
                        "selected_round": round_idx,
                        "prompt_name": item["prompt_name"],
                        "mutation": item["mutation"],
                        "rouge1": item["rouge1"],
                        "prompt_text": item["prompt_text"]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        population = [{"name": s["prompt_name"], "text": s["prompt_text"]} for s in selected]

if __name__ == "__main__":
    with open("data/cnn_input.json", "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open("data/xsum_input.json", "r", encoding="utf-8") as f:
        xsum_data = json.load(f)

    datasets = {
        "cnn": [(item["article"], item["reference"]) for item in cnn_data],
        "xsum": [(item["article"], item["reference"]) for item in xsum_data]
    }

    INITIAL_PROMPTS = [
        {"name": "zero_shot", "text": "Can you give me a short summary of this article?"},
        {"name": "few_shot", "text": """Here are some examples of articles and their summaries:

        Example 1:
        Article: A storm hit northern France yesterday, damaging homes and cutting power.
        Summary: A storm in northern France caused damage and power outages.

        Example 2:
        Article: A new study shows coffee may help reduce the risk of heart disease.
        Summary: Researchers found coffee drinkers had lower heart disease risk.

        Now here is a new article. Please write its summary:"""},
        {"name": "instruction_based", "text": "Please explain the main points of this article briefly in three sentences: "},
        {"name": "pattern_based", "text": "Can you list the three most important facts from this article as bullet points?"},
        {"name": "target_audience", "text": "Can you rewrite this article so that it’s easy to understand for everyday readers?"}
    ]
    run_evolution(datasets, INITIAL_PROMPTS, max_rounds=5, top_k=5, rouge1_threshold=0.5)
