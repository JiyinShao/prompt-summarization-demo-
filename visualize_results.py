import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULT_DIR = "results"

MUT_ABBR = {
    "synonym_replacement": "SR",
    "prompt_rewriting": "PR",
    "style_instruction": "SI",
    "audience_information": "AI",
    "stepwise_prompt": "SP",
    "none": "none"
}

def shorten_mutation(name: str) -> str:
    return MUT_ABBR.get(name, name)

def plot_rounds():
    files = [f for f in os.listdir(RESULT_DIR) if f.startswith("round_") and f.endswith(".json") and "remain" not in f]
    for f in sorted(files):
        path = os.path.join(RESULT_DIR, f)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        df = pd.DataFrame(data)
        if df.empty: 
            continue
        df["mutation_short"] = df["mutation"].apply(shorten_mutation)
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="prompt_name", y="rouge1", hue="mutation_short", errorbar=None)
        plt.xticks(rotation=0)
        plt.title(f"{f} — rouge1 by prompt & mutation (mean)")
        plt.savefig(os.path.join(RESULT_DIR, f.replace(".json","_bar.png")))
        plt.close()

def plot_threshold_trend():
    path = os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(path): 
        return
    records = []
    with open(path,"r",encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if "meet_round" not in df: 
        return
    trend = df.groupby("meet_round").size().reset_index(name="count")
    plt.figure(figsize=(8,5))
    plt.plot(trend["meet_round"], trend["count"], marker="o")
    plt.xticks(trend["meet_round"].unique())
    plt.title("Meet Threshold Trend")
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.savefig(os.path.join(RESULT_DIR,"threshold_trend.png"))
    plt.close()

def plot_threshold_stats():
    path = os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(path):
        return
    records = []
    with open(path,"r",encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty: 
        return
    df["mutation_short"] = df["mutation"].apply(shorten_mutation)
    plt.figure(figsize=(8,5))
    df["prompt_name"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Prompt Count (Meet Threshold)")
    plt.savefig(os.path.join(RESULT_DIR,"threshold_prompts.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    df["mutation_short"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Mutation Count (Meet Threshold)")
    plt.savefig(os.path.join(RESULT_DIR,"threshold_mutations.png"))
    plt.close()

def plot_topk_stats():
    path = os.path.join(RESULT_DIR,"selected_topk.json")
    if not os.path.exists(path):
        return
    records = []
    with open(path,"r",encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty: 
        return
    df["mutation_short"] = df["mutation"].apply(shorten_mutation)
    plt.figure(figsize=(8,5))
    df["prompt_name"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Prompt Count (Selected Top-k)")
    plt.savefig(os.path.join(RESULT_DIR,"topk_prompts.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    df["mutation_short"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.title("Mutation Count (Selected Top-k)")
    plt.savefig(os.path.join(RESULT_DIR,"topk_mutations.png"))
    plt.close()

def plot_rouge_vs_compression():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty:
        return
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="compression", y="rouge1", hue="prompt_name", style="mutation", s=80)
    plt.title("Relationship between ROUGE-1 and Compression Rate")
    plt.xlabel("Compression Rate")
    plt.ylabel("ROUGE-1 Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "rouge_vs_compression.png"))
    plt.close()

def plot_rouge_vs_compression_trend():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    if df.empty:
        return
    sns.lmplot(data=df, x="compression", y="rouge1", hue="prompt_name", aspect=1.2, height=6, scatter_kws={"s":40}, ci=None)
    plt.title("ROUGE-1 vs Compression (Regression Trend by Prompt)")
    plt.xlabel("Compression Rate")
    plt.ylabel("ROUGE-1")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "rouge_vs_compression_trend.png"))
    plt.close()

def plot_score_distributions():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(records)
    if df.empty:
        print("No data loaded."); return

    df["fre"] = df["fre"] / 100.0
    metrics = ["rouge1", "rougel", "fre", "compression"]
    plt.figure(figsize=(10,5))
    melted = df[metrics].melt(var_name="Metric", value_name="Score")
    sns.boxplot(data=melted, x="Metric", y="Score", hue="Metric", palette="tab10", legend=False)

    plt.title("Score Distribution by Metric")
    plt.ylabel("Normalized Score (FRE ÷ 100)")
    plt.xlabel("Metric")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = os.path.join(RESULT_DIR, "score_distributions.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_metric_relationships():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(records)
    if df.empty:
        return
    sns.pairplot(df, vars=["rouge1", "rougel", "fre", "compression"], hue="prompt_name")
    plt.suptitle("Metric Relationships", y=1.02)
    plt.savefig(os.path.join(RESULT_DIR, "metric_relationships.png"))
    plt.close()

def plot_prompt_radar():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(records)
    if df.empty:
        return
    agg = df.groupby("prompt_name").agg({
        "rouge1":"mean","rougel":"mean","fre":"mean","compression":"mean"
    }).reset_index()
    agg["fre_norm"] = agg["fre"]/100.0
    agg["brevity"] = 1.0 - agg["compression"]
    metrics = ["rouge1","rougel","fre_norm","brevity"]
    labels = ["rouge1","rougel","fre","compression"]
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    for _, row in agg.iterrows():
        vals = [row[m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1, label=row["prompt_name"])
        ax.fill(angles, vals, alpha=0.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Average Performance per Prompt (normalized)")
    ax.set_ylim(0,1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35,1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,"prompt_radar.png"), dpi=180)
    plt.close()

def plot_dataset_bar():
    path = os.path.join(RESULT_DIR, "round_5.json")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    if df.empty:
        print("No data loaded."); return

    summary = df.groupby("dataset")[["rouge1", "rougel", "fre"]].mean().round(3)
    fig, ax1 = plt.subplots(figsize=(8,5))
    width = 0.25
    x = range(len(summary.index))
    ax1.bar([i - width for i in x], summary["rouge1"], width=width, label="ROUGE-1")
    ax1.bar(x, summary["rougel"], width=width, label="ROUGE-L")
    ax2 = ax1.twinx()
    ax2.bar([i + width for i in x], summary["fre"], width=width, color="green", label="FRE")

    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("ROUGE Score")
    ax2.set_ylabel("Flesch Reading Ease")
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary.index, rotation=0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.set_title("Comparison of Evaluation Metrics between CNN and XSUM")

    for i, val in enumerate(summary["rouge1"]):
        ax1.text(i - width, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    for i, val in enumerate(summary["rougel"]):
        ax1.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    for i, val in enumerate(summary["fre"]):
        ax2.text(i + width, val + 2, f"{val:.1f}", ha="center", fontsize=9, color="green")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, "dataset_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close() 

def plot_baseline_vs_final():
    RESULT_DIR="results"
    baseline_path=os.path.join(RESULT_DIR,"baseline.json")
    final_path=os.path.join(RESULT_DIR,"round_5.json")
    meet_path=os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(baseline_path):
        print("Missing baseline.json")
        return
    if not os.path.exists(final_path) and not os.path.exists(meet_path):
        print("Missing round_5.json and meet_threshold.json")
        return
    with open(baseline_path,"r",encoding="utf-8") as f:
        baseline_data=json.load(f)
    final_data=[]
    if os.path.exists(final_path):
        with open(final_path,"r",encoding="utf-8") as f:
            final_data.extend(json.load(f))
    if os.path.exists(meet_path):
        with open(meet_path,"r",encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item=json.loads(line)
                final_data.append(item)
    if not final_data:
        print("No final data found")
        return
    df_base=pd.DataFrame(baseline_data)
    df_final=pd.DataFrame(final_data)
    if df_base.empty or df_final.empty:
        print("Empty result file")
        return
    base_avg=df_base[["rouge1","rougel","fre"]].mean().round(3)
    final_avg=df_final[["rouge1","rougel","fre"]].mean().round(3)
    summary=pd.DataFrame({"ROUGE-1":[base_avg["rouge1"],final_avg["rouge1"]],"ROUGE-L":[base_avg["rougel"],final_avg["rougel"]],"FRE":[base_avg["fre"],final_avg["fre"]]},index=["Baseline","Final"])
    fig,ax1=plt.subplots(figsize=(8,5))
    ax1.plot(summary.index,summary["ROUGE-1"],marker="o",label="ROUGE-1")
    ax1.plot(summary.index,summary["ROUGE-L"],marker="s",label="ROUGE-L")
    ax2=ax1.twinx()
    ax2.plot(summary.index,summary["FRE"],marker="^",color="green",label="FRE")
    ax1.set_xlabel("Model Stage")
    ax1.set_ylabel("ROUGE Score")
    ax2.set_ylabel("Flesch Reading Ease")
    ax1.set_xticks(range(len(summary.index)))
    ax1.set_xticklabels(summary.index,rotation=0)
    ax1.grid(axis="y",linestyle="--",alpha=0.5)
    ax1.set_title("Baseline vs Final (All Meet-Threshold)")
    for i,val in enumerate(summary["ROUGE-1"]):
        ax1.text(i,val,f"{val:.2f}",ha="center",va="bottom",fontsize=9)
    for i,val in enumerate(summary["ROUGE-L"]):
        ax1.text(i,val,f"{val:.2f}",ha="center",va="bottom",fontsize=9)
    for i,val in enumerate(summary["FRE"]):
        ax2.text(i,val,f"{val:.1f}",ha="center",va="bottom",fontsize=9,color="green")
    h1,l1=ax1.get_legend_handles_labels()
    h2,l2=ax2.get_legend_handles_labels()
    ax1.legend(h1+h2,l1+l2,loc="upper left",fontsize=9)
    plt.tight_layout()
    save_path=os.path.join(RESULT_DIR,"baseline_vs_final.png")
    plt.savefig(save_path,dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_rounds()
    plot_threshold_trend()
    plot_threshold_stats()
    plot_topk_stats()
    plot_rouge_vs_compression()
    plot_rouge_vs_compression_trend()
    plot_score_distributions()
    plot_metric_relationships()
    plot_prompt_radar()
    plot_dataset_bar()
    plot_baseline_vs_final()