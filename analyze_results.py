import os, glob, json
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

files = glob.glob("results/*_prompt_eval_*.json")
records = []
for f in files:
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        records.extend(data)

df = pd.DataFrame(records)
df.to_csv("results/all_results.csv", index=False)

metrics = ["rouge1", "rougeL", "fre"]
grouped = df.groupby(["dataset", "template"])[metrics].mean().reset_index()

for m in metrics:
    pivot = grouped.pivot(index="template", columns="dataset", values=m).sort_index()
    ax = pivot.plot(kind="bar", figsize=(9,5))
    ax.set_xlabel("template")
    ax.set_ylabel(m)
    ax.set_title(f"{m} by template and dataset")
    plt.tight_layout()
    plt.savefig(f"results/{m}_by_template_dataset.png")
    plt.close()

overall = df.groupby("template")[metrics].mean().reset_index()
overall.to_csv("results/mean_by_template.csv", index=False)

ax = overall.set_index("template")[metrics].plot(kind="bar", figsize=(9,5))
ax.set_xlabel("template")
ax.set_ylabel("score")
ax.set_title("mean scores by template (averaged over datasets)")
plt.tight_layout()
plt.savefig("results/mean_scores_by_template.png")
plt.close()
