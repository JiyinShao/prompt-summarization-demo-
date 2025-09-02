import os, glob, json, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- short aliases for cleaner legends ----
ALIAS = {
    "synonym_replacement": "SR",
    "prompt_rewriting": "PR",
    "style_instruction": "SI",
    "audience_information": "AI",
    "stepwise_prompt": "SP",
    "none": "NONE",
}

BASE_ORDER = ["SR", "PR", "SI", "AI", "SP"]  # for component frequency plot

def round_num_from_name(path: str) -> int:
    m = re.search(r"round(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 0

def alias_chain(m: str) -> str:
    if not isinstance(m, str) or not m:
        return "NONE"
    parts = [ALIAS.get(p.strip(), p.strip()) for p in m.split("+")]
    return "+".join(parts)

# ========== Part 1: per-round charts ==========
# x = template, colors = (aliased) mutation, score = ROUGE-1 (fallback ROUGE-L)
round_files = sorted(
    [p for p in glob.glob(os.path.join(RESULT_DIR, "round*.json"))
     if ("selected" not in p and "passed" not in p)],
    key=round_num_from_name
)

for rf in round_files:
    with open(rf, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
    df = pd.DataFrame(data)
    if df.empty or "template" not in df or "mutation" not in df:
        continue

    metric = "rouge1" if "rouge1" in df.columns else "rougeL"
    df = df.dropna(subset=[metric])

    df["mutation_alias"] = df["mutation"].apply(alias_chain)

    g = df.groupby(["template","mutation_alias"], as_index=False)[metric].mean()

    templates = sorted(g["template"].unique().tolist())
    muts = sorted(g["mutation_alias"].unique().tolist())

    mat = np.full((len(templates), len(muts)), np.nan, dtype=float)
    idx_t = {t:i for i,t in enumerate(templates)}
    idx_m = {m:i for i,m in enumerate(muts)}

    for _, row in g.iterrows():
        i = idx_t[row["template"]]
        j = idx_m[row["mutation_alias"]]
        mat[i, j] = row[metric]

    x = np.arange(len(templates))
    w = 0.8 / max(1, len(muts))

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, m in enumerate(muts):
        vals = mat[:, j]
        ax.bar(x + (j - (len(muts)-1)/2)*w, vals, width=w, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(templates, rotation=18, ha="right")
    ax.set_ylabel(metric)
    ax.set_xlabel("template")
    ax.set_title(f"{os.path.basename(rf).replace('.json','')} — {metric} by template & mutation")
    # put legend outside to avoid clutter
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, ncol=1, title="mutation")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_png = os.path.join(RESULT_DIR, f"{os.path.basename(rf).replace('.json','')}_{metric}_by_template_mutation.png")
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

# ========== Part 2: selection frequency ==========
# 2a) template frequency across rounds (count how often it appears in round*_selected.json)
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

    # --- template frequency (how often a template is selected) ---
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

    # --- mutation component frequency (split chains, ignore NONE) ---
    if "mutation" not in sdf.columns:
        raise SystemExit("round*_selected.json must contain 'mutation'.")

    # explode chain into components, alias to SR/PR/SI/AI/SP, drop NONE
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
    # order by BASE_ORDER if present
    freq_comp["order"] = freq_comp["component"].apply(lambda x: BASE_ORDER.index(x) if x in BASE_ORDER else 999)
    freq_comp = freq_comp.sort_values(["order","count"], ascending=[True, False]).drop(columns=["order"])
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

    # keep a tidy per-round table for inspection
    sdf_out = sdf[["round_num","template","mutation"]].copy()
    sdf_out["mutation_alias"] = sdf_out["mutation"].apply(alias_chain)
    sdf_out.sort_values(["round_num","template","mutation_alias"]).to_csv(
        os.path.join(RESULT_DIR, "selected_pairs_by_round.csv"), index=False
    )
else:
    print("[info] no round*_selected.json found; skipping frequency plots.")

print("[done] minimal evolution analysis completed.")
