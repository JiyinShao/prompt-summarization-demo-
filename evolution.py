import os, json
import pandas as pd

from config import (
    DATASETS, SAMPLES_PER_DATASET, PROMPT_TEMPLATES, ACTIVE_MUTATIONS,
    MAX_ROUNDS, TOP_K, THRESHOLDS, MUTATION_POOL, RESULT_DIR
)
from main import run_once, load_samples

os.makedirs(RESULT_DIR, exist_ok=True)

def all_candidates():
    return [(t, m) for t in PROMPT_TEMPLATES for m in ACTIVE_MUTATIONS]

def passed_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df["rouge1"] >= THRESHOLDS["rouge1"]) & (df["fre"] >= THRESHOLDS["fre"])
    return df.loc[mask].copy()

def select_top_k(df: pd.DataFrame, k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["template","mutation"])
    g = df.groupby(["template","mutation"], as_index=False)[["rouge1","rougeL","fre"]].mean()
    g = g.sort_values(["rouge1","rougeL","fre"], ascending=False)
    return g.head(k)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def stack_pairs(top_df: pd.DataFrame) -> list[tuple[str, str]]:
    pairs = []
    preferred = ["style_instruction", "audience_information", "prompt_rewriting", "synonym_replacement", "stepwise_prompt"]
    for _, row in top_df.iterrows():
        t = str(row["template"])
        cur = str(row["mutation"])
        used = set(cur.split("+"))
        for m in preferred:
            if m not in used and m in MUTATION_POOL:
                pairs.append((t, f"{cur}+{m}"))
                break
        else:
            pairs.append((t, cur))
    seen, uniq = set(), []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq

def run_evolution():
    fixed_samples = {ds: load_samples(ds, SAMPLES_PER_DATASET) for ds in DATASETS}

    candidates = all_candidates()
    for r in range(1, MAX_ROUNDS + 1):
        print(f"\n=== Round {r} | candidates: {len(candidates)} ===")
        results = run_once(
            datasets=DATASETS,
            samples_per_dataset=SAMPLES_PER_DATASET,
            candidates=candidates,
            samples_map=fixed_samples, 
        )
        round_path = os.path.join(RESULT_DIR, f"round{r}.json")
        save_json(results, round_path)
        print(f"[saved] {round_path}")

        df = pd.DataFrame(results)
        if df.empty:
            print("[stop] no results, exiting.")
            return
        for col in ["rouge1","rougeL","fre"]:
            if col not in df.columns:
                df[col] = None
        df = df.dropna(subset=["rouge1","rougeL","fre"])

        ok = passed_df(df)
        if not ok.empty:
            save_json(ok.to_dict(orient="records"), os.path.join(RESULT_DIR, f"round{r}_passed.json"))
            print(f"[stop] threshold reached in round {r}")
            return

        top = select_top_k(df, TOP_K)
        save_json(top.to_dict(orient="records"), os.path.join(RESULT_DIR, f"round{r}_selected.json"))
        candidates = stack_pairs(top)
        if not candidates:
            print("[stop] no candidates for next round")
            return

    print(f"[stop] reached MAX_ROUNDS={MAX_ROUNDS}")

if __name__ == "__main__":
    run_evolution()
