import json
import os
from llm_utils import summarize_with_prompt
from evaluation import evaluate_summary

def run_evolution(datasets, prompt_file, round_idx=1, rouge1_threshold=0.5, top_k=5):
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)
    meet_file = os.path.join(RESULT_DIR, "meet_threshold.json")
    selected_file = os.path.join(RESULT_DIR, "selected_topk.json")
    open(meet_file, "a").close()
    open(selected_file, "a").close()

    with open(prompt_file, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    print(f"\n=== Round {round_idx} | candidates={len(candidates)} ===")
    results = []
    total = sum(len(samples) * len(candidates) for samples in datasets.values())
    count = 0

    for dataset_name, samples in datasets.items():
        for item in samples:
            article_id = item["id"]
            article = item["article"]
            ref = item["reference"]
            for cand in candidates:
                parent_name = cand.get("parent_name", cand.get("name", f"round{round_idx}_unknown"))
                mutation = cand.get("mutation", "none")
                summary = summarize_with_prompt(article, cand["text"] if "text" in cand else cand["prompt_text"])
                scores = evaluate_summary(summary, ref, article)
                record = {
                    "round": round_idx,
                    "dataset": dataset_name,
                    "article_id": article_id,
                    "parent_name": parent_name,
                    "mutation": mutation,
                    "prompt_text": cand["text"] if "text" in cand else cand["prompt_text"],
                    "summary": summary,
                    "rouge1": scores["rouge1"],
                    "rougel": scores["rougel"],
                    "fre": scores["fre"],
                    "compression": scores["compression"]
                }
                results.append(record)
                count += 1
                if count % 10 == 0:
                    print(f"[Round {round_idx}] Progress: {count}/{total} ({count/total:.1%})")

    with open(os.path.join(RESULT_DIR, f"round_{round_idx}_results.json"), "w", encoding="utf-8") as f:
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
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    next_round_prompts = []
    for s in selected:
        next_round_prompts.append({
            "name": s["parent_name"],
            "text": s["prompt_text"]
        })

    with open(f"data/round_{round_idx+1}_prompts.json", "w", encoding="utf-8") as f:
        json.dump(next_round_prompts, f, ensure_ascii=False, indent=2)

    print(f"Top {top_k} prompts saved for next round → data/round_{round_idx+1}_prompts.json")


if __name__ == "__main__":
    with open("data/cnn_input.json", "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open("data/xsum_input.json", "r", encoding="utf-8") as f:
        xsum_data = json.load(f)

    datasets = {
        "cnn": [{"id": item["id"], "article": item["article"], "reference": item["reference"]} for item in cnn_data],
        "xsum": [{"id": item["id"], "article": item["article"], "reference": item["reference"]} for item in xsum_data]
    }

    run_evolution(datasets, "results/round_1_prompts.json", round_idx=1)
