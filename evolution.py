import os
import json
import pandas as pd

from config import (
    DATASETS,
    SAMPLES_PER_DATASET,
    PROMPT_TEMPLATES,
    ACTIVE_MUTATIONS,
    MAX_ROUNDS,
    TOP_K,
    MUTATION_POOL,
    RESULT_DIR,
)
from main import run_once, load_samples

os.makedirs(RESULT_DIR, exist_ok=True)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_chain(chain: str):
    return [] if not chain else chain.split(" + ")

def ends_with_none(chain: str) -> bool:
    parts = parse_chain(chain)
    return len(parts) > 0 and parts[-1] == "none"

def all_grid_candidates():
    return [(t, m) for t in PROMPT_TEMPLATES for m in ACTIVE_MUTATIONS]

def expand_6(template: str, chain: str):
    used = set(parse_chain(chain))
    out = []

    out.append((template, (chain + " + none") if chain else "none"))

    for m in MUTATION_POOL:
        if m == "none":
            continue
        if m in used:
            continue
        new_m = (chain + " + " + m) if chain else m
        out.append((template, new_m))
    return out

def build_round2plus(parents_prev):
    cand = set()
    for t, m in parents_prev:
        if ends_with_none(m):
            cand.add((t, m))
        else:
            cand.update(expand_6(t, m))
    return list(cand)

def group_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["rouge1", "rougeL", "fre", "compression_ratio"]:
        if col not in df.columns:
            df[col] = None
    g = (
        df.groupby(["template", "mutation"], as_index=False)[
            ["rouge1", "rougeL", "fre", "compression_ratio"]
        ]
        .mean()
        .fillna({"rouge1": 0.0, "rougeL": 0.0, "fre": 0.0, "compression_ratio": 0.0})
        .sort_values(["rouge1", "rougeL", "fre"], ascending=False, kind="mergesort")
    )
    return g

def top_k_pairs(df_scores: pd.DataFrame, k: int) -> pd.DataFrame:
    if df_scores.empty:
        return df_scores
    return df_scores.head(k)[["template", "mutation"]].copy()

def all_none_selected(top_df: pd.DataFrame) -> bool:
    if top_df.empty:
        return False
    return all(ends_with_none(m) for m in top_df["mutation"].tolist())

def load_fixed_inputs():
    fixed = {}
    for ds in DATASETS:
        p = os.path.join(RESULT_DIR, f"{ds}_input.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                items = json.load(f)
            fixed[ds] = items[: SAMPLES_PER_DATASET]
        else:
            fixed[ds] = load_samples(ds, SAMPLES_PER_DATASET)
    return fixed

def run_evolution():
    fixed_samples = load_fixed_inputs()

    candidates = all_grid_candidates()

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
            print("[stop] no results")
            return

        scores = group_scores(df)
        scores_path = os.path.join(RESULT_DIR, f"round{r}_scores.json")
        save_json(scores.to_dict(orient="records"), scores_path)

        top_df = top_k_pairs(scores, TOP_K)
        sel_path = os.path.join(RESULT_DIR, f"round{r}_selected.json")
        save_json(top_df.to_dict(orient="records"), sel_path)
        print(f"[next] select top-{TOP_K} -> {sel_path}")

        if all_none_selected(top_df):
            print(f"[stop] all selected pairs end with 'none' in round {r}")
            return

        if r == MAX_ROUNDS:
            print(f"[stop] reached MAX_ROUNDS={MAX_ROUNDS}")
            return

        parents_prev = list(
            set(tuple(x) for x in top_df.itertuples(index=False, name=None))
        )
        candidates = build_round2plus(parents_prev)

if __name__ == "__main__":
    run_evolution()
