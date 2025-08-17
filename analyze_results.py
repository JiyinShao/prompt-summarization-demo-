import os, glob, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

files = sorted(glob.glob("results/*_prompt_eval_*.json"))
records = []
for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        records.extend(data)

df = pd.DataFrame(records)
if "mutation" not in df.columns:
    df["mutation"] = "none"

df.to_csv("results/all_results.csv", index=False)

metrics = ["rouge1", "rougeL", "fre"]

g_dt = df.groupby(["dataset","template"])[metrics].mean().reset_index()
g_dt.to_csv("results/mean_by_dataset_template.csv", index=False)

g_t = df.groupby(["template"])[metrics].mean().reset_index()
g_t.to_csv("results/mean_by_template.csv", index=False)

has_mut = df["mutation"].nunique() > 1
if has_mut:
    g_dtm = df.groupby(["dataset","template","mutation"])[metrics].mean().reset_index()
    g_dtm.to_csv("results/mean_by_dataset_template_mutation.csv", index=False)
    g_tm = df.groupby(["template","mutation"])[metrics].mean().reset_index()
    g_tm.to_csv("results/mean_by_template_mutation.csv", index=False)

for m in metrics:
    pivot = g_dt.pivot(index="template", columns="dataset", values=m).sort_index()
    ax = pivot.plot(kind="bar", figsize=(9,5))
    ax.set_xlabel("template")
    ax.set_ylabel(m)
    ax.set_title(f"{m} by template and dataset")
    plt.tight_layout()
    plt.savefig(f"results/{m}_by_template_dataset.png")
    plt.close()

fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
templates = g_t["template"].tolist()
x = np.arange(len(templates))
w = 0.27
ax1.bar(x - w, g_t["rouge1"], width=w, label="rouge1", color="#1f77b4")
ax1.bar(x,      g_t["rougeL"], width=w, label="rougeL", color="#ff7f0e")
ax2.bar(x + w,  g_t["fre"],    width=w, label="fre",    color="#2ca02c", alpha=0.75)
ax1.set_xticks(x)
ax1.set_xticklabels(templates, rotation=25, ha="right")
ax1.set_xlabel("template")
ax1.set_ylabel("ROUGE (0–1)")
ax2.set_ylabel("FRE (unbounded)")
ax1.set_title("Mean scores by template (ROUGE vs FRE)")
h1,l1 = ax1.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc="upper left")
plt.tight_layout()
plt.savefig("results/mean_scores_by_template_dual_axis.png")
plt.close()

if has_mut:
    for m in metrics:
        p = g_tm.pivot(index="template", columns="mutation", values=m).reindex(templates)
        ax = p.plot(kind="bar", figsize=(11,5))
        ax.set_xlabel("template")
        ax.set_ylabel(m)
        ax.set_title(f"{m} by template and mutation (avg over datasets)")
        plt.tight_layout()
        plt.savefig(f"results/{m}_by_template_mutation.png")
        plt.close()

    for m in metrics:
        for ds in df["dataset"].unique():
            sub = g_dtm[g_dtm["dataset"]==ds]
            p = sub.pivot(index="template", columns="mutation", values=m).reindex(templates)
            ax = p.plot(kind="bar", figsize=(11,5))
            ax.set_xlabel("template")
            ax.set_ylabel(m)
            ax.set_title(f"{m} by template and mutation — {ds}")
            plt.tight_layout()
            plt.savefig(f"results/{m}_by_template_mutation_{ds}.png")
            plt.close()
