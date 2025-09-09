import os, glob, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

ALIAS = {
    "synonym_replacement": "SR",
    "prompt_rewriting": "PR",
    "style_instruction": "SI",
    "audience_information": "AI",
    "stepwise_prompt": "SP",
    "none": "NONE",
}
BASE_ORDER = ["SR", "PR", "SI", "AI", "SP"]
MAX_MUTS_PER_FIG = 12 

def round_num_from_name(path: str) -> int:
    m = re.search(r"round(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 0

def alias_chain(m: str) -> str:
    if not isinstance(m, str) or not m:
        return "NONE"
    parts = [ALIAS.get(p.strip(), p.strip()) for p in m.split("+")]
    return "+".join(parts)

def load_round_scores(round_json: str) -> pd.DataFrame:
    base = os.path.basename(round_json)
    rnum = round_num_from_name(base)
    scores_path = os.path.join(RESULT_DIR, f"round{rnum}_scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, dict):
                data = [data]
        df = pd.DataFrame(data)
        need_cols = {"template", "mutation", "rouge1", "rougeL", "fre"}
        for c in need_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    with open(round_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
    df = pd.DataFrame(data)
    if df.empty:
        return df
    for c in ["rouge1", "rougeL", "fre"]:
        if c not in df.columns:
            df[c] = np.nan
    g = (
        df.groupby(["template", "mutation"], as_index=False)[["rouge1", "rougeL", "fre"]]
        .mean()
        .fillna({"rouge1": 0.0, "rougeL": 0.0, "fre": 0.0})
    )
    return g

round_files = sorted(
    [p for p in glob.glob(os.path.join(RESULT_DIR, "round*.json"))
     if ("selected" not in p and "passed" not in p and "scores" not in p)],
    key=round_num_from_name
)

for rf in round_files:
    df = load_round_scores(rf)
    if df.empty or "template" not in df or "mutation" not in df:
        continue

    metric = "rouge1" if "rouge1" in df.columns else "rougeL"
    df = df.dropna(subset=[metric])

    df["mutation_alias"] = df["mutation"].astype(str).apply(alias_chain)

    top_muts = (
        df.groupby("mutation_alias", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
        .head(MAX_MUTS_PER_FIG)["mutation_alias"]
        .tolist()
    )
    df = df[df["mutation_alias"].isin(top_muts)]

    g = (
        df.groupby(["template", "mutation_alias"], as_index=False)[metric]
        .mean()
        .sort_values(["template", metric], ascending=[True, False])
    )

    templates = sorted(g["template"].unique().tolist())
    muts = sorted(g["mutation_alias"].unique().tolist())

    mat = np.full((len(templates), len(muts)), np.nan, dtype=float)
    idx_t = {t: i for i, t in enumerate(templates)}
    idx_m = {m: i for i, m in enumerate(muts)}

    for _, row in g.iterrows():
        i = idx_t[row["template"]]
        j = idx_m[row["mutation_alias"]]
        mat[i, j] = row[metric]

    x = np.arange(len(templates))
    w = 0.8 / max(1, len(muts))

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, m in enumerate(muts):
        vals = mat[:, j]
        ax.bar(x + (j - (len(muts) - 1) / 2) * w, vals, width=w, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(templates, rotation=18, ha="right")
    ax.set_ylabel(metric)
    ax.set_xlabel("template")
    title = f"{os.path.basename(rf).replace('.json','')} — {metric} by template & mutation (top-{MAX_MUTS_PER_FIG} muts)"
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, ncol=1, title="mutation")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_png = os.path.join(
        RESULT_DIR,
        f"{os.path.basename(rf).replace('.json','')}_{metric}_by_template_mutation.png"
    )
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

selected_files = sorted(glob.glob(os.path.join(RESULT_DIR, "round*_selected.json")), key=round_num_from_name)

sel_records = []
for sf in selected_files:
    with open(sf, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
    rnum = round_num_from_name(sf)
    for d in data:
        d = dict(d)
        d["round_num"] = rnum
        sel_records.append(d)

if sel_records:
    sdf = pd.DataFrame(sel_records)

    if "template" not in sdf.columns:
        raise SystemExit("round*_selected.json must contain 'template'.")

    freq_t = sdf["template"].value_counts().rename_axis("template").reset_index(name="count")
    freq_t.to_csv(os.path.join(RESULT_DIR, "selected_frequency_by_template.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(freq_t["template"], freq_t["count"])
    ax.set_xlabel("template")
    ax.set_ylabel("selected count")
    ax.set_title("Selection frequency by template (across rounds)")
    ax.set_xticklabels(freq_t["template"], rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "selected_frequency_by_template.png"), bbox_inches="tight")
    plt.close()

    if "mutation" not in sdf.columns:
        raise SystemExit("round*_selected.json must contain 'mutation'.")

    comp = []
    for m in sdf["mutation"].astype(str):
        parts = [p.strip() for p in m.split("+")] if m else ["none"]
        parts = [ALIAS.get(p, p) for p in parts]
        parts = [p for p in parts if p != "NONE"]
        if not parts:
            continue
        comp.extend(parts)

    comp_series = pd.Series(comp, name="component")
    freq_comp = comp_series.value_counts().rename_axis("component").reset_index(name="count")
    freq_comp["order"] = freq_comp["component"].apply(lambda x: BASE_ORDER.index(x) if x in BASE_ORDER else 999)
    freq_comp = freq_comp.sort_values(["order", "count"], ascending=[True, False]).drop(columns=["order"])
    freq_comp.to_csv(os.path.join(RESULT_DIR, "selected_frequency_by_mutation_component.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(freq_comp["component"], freq_comp["count"])
    ax.set_xlabel("mutation component (SR/PR/SI/AI/SP)")
    ax.set_ylabel("selected count")
    ax.set_title("Selection frequency by mutation component (across rounds)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "selected_frequency_by_mutation.png"), bbox_inches="tight")
    plt.close()

    sdf_out = sdf[["round_num", "template", "mutation"]].copy()
    sdf_out["mutation_alias"] = sdf_out["mutation"].apply(alias_chain)
    sdf_out.sort_values(["round_num", "template", "mutation_alias"]).to_csv(
        os.path.join(RESULT_DIR, "selected_pairs_by_round.csv"), index=False
    )
else:
    print("[info] no round*_selected.json found; skipping frequency plots.")

print("[done] evolution analysis updated.")
