import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULT_DIR = "results"
PIC_DIR = os.path.join(RESULT_DIR, "pic")
os.makedirs(PIC_DIR, exist_ok=True)

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

def load_and_flatten(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for d in data:
        if isinstance(d.get("rouge"), dict):
            d["rouge1"] = d["rouge"].get("rouge1")
            d["rougel"] = d["rouge"].get("rougel")
            d.pop("rouge", None)
    return pd.DataFrame(data)

def plot_rounds():
    files = [f for f in os.listdir(RESULT_DIR) if f.startswith("round_") and f.endswith("_results.json")]
    for f in sorted(files):
        path = os.path.join(RESULT_DIR, f)
        df = load_and_flatten(path)
        if df.empty or "rouge1" not in df.columns or "parent_name" not in df.columns:
            continue
        df["mutation_short"] = df["mutation"].apply(shorten_mutation)
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="parent_name", y="rouge1", hue="mutation_short", errorbar=None)
        plt.xticks(rotation=0)
        plt.title(f"{f} — rouge1 by prompt & mutation (mean)")
        plt.savefig(os.path.join(PIC_DIR, f.replace(".json","_bar.png")))
        plt.close()

def plot_threshold_trend():
    path = os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(path): 
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if "meet_round" not in df: 
        return
    trend = df.groupby("meet_round").size().reset_index(name="count")
    plt.figure(figsize=(8,5))
    plt.plot(trend["meet_round"], trend["count"], marker="o")
    plt.xticks(trend["meet_round"].unique())
    plt.xlabel("Round")
    plt.ylabel("Count")
    plt.savefig(os.path.join(PIC_DIR,"threshold_trend.png"))
    plt.close()

def plot_threshold_stats():
    path = os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "parent_name" not in df.columns: 
        return
    df["mutation_short"] = df["mutation"].apply(shorten_mutation)
    plt.figure(figsize=(8,5))
    df["parent_name"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(PIC_DIR,"threshold_prompts.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    df["mutation_short"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(PIC_DIR,"threshold_mutations.png"))
    plt.close()

def plot_topk_stats():
    path = os.path.join(RESULT_DIR,"selected_topk.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "parent_name" not in df.columns: 
        return
    df["mutation_short"] = df["mutation"].apply(shorten_mutation)
    plt.figure(figsize=(8,5))
    df["parent_name"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(PIC_DIR,"topk_prompts.png"))
    plt.close()
    plt.figure(figsize=(8,5))
    df["mutation_short"].value_counts().plot(kind="bar")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(PIC_DIR,"topk_mutations.png"))
    plt.close()

def plot_rouge_vs_compression():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "rouge1" not in df.columns or "parent_name" not in df.columns:
        return
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="compression", y="rouge1", hue="parent_name", style="mutation", s=80)
    plt.xlabel("Compression Rate")
    plt.ylabel("ROUGE-1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, "rouge_vs_compression.png"), dpi=300)
    plt.close()

def plot_rouge_vs_compression_trend():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "rouge1" not in df.columns or "parent_name" not in df.columns:
        return
    sns.lmplot(data=df, x="compression", y="rouge1", hue="parent_name", aspect=1.2, height=6, scatter_kws={"s":40}, ci=None)
    plt.xlabel("Compression Rate")
    plt.ylabel("ROUGE-1")
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, "rouge_vs_compression_trend.png"))
    plt.close()

def plot_score_distributions():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "rouge1" not in df.columns:
        return
    df["fre"] = df["fre"] / 100.0
    metrics = ["rouge1", "rougel", "fre", "compression"]
    plt.figure(figsize=(10,5))
    melted = df[metrics].melt(var_name="Metric", value_name="Score")
    sns.boxplot(data=melted, x="Metric", y="Score", hue="Metric", palette="tab10", legend=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, "score_distributions.png"), dpi=300)
    plt.close()

def plot_metric_relationships(): 
    path = os.path.join(RESULT_DIR, "meet_threshold.json") 
    if not os.path.exists(path): 
        return 
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()] 
    df = pd.DataFrame(records) 
    if df.empty or "rouge1" not in df.columns or "parent_name" not in df.columns: 
        return 
    sns.pairplot(df, vars=["rouge1", "rougel", "fre", "compression"], hue="parent_name") 
    plt.savefig(os.path.join(PIC_DIR, "metric_relationships.png")) 
    plt.close()

def plot_prompt_radar():
    path = os.path.join(RESULT_DIR, "meet_threshold.json")
    if not os.path.exists(path):
        return
    records = [json.loads(line) for line in open(path,"r",encoding="utf-8") if line.strip()]
    df = pd.DataFrame(records)
    if df.empty or "rouge1" not in df.columns or "parent_name" not in df.columns:
        return
    agg = df.groupby("parent_name").agg({"rouge1":"mean","rougel":"mean","fre":"mean","compression":"mean"}).reset_index()
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
        ax.plot(angles, vals, linewidth=1, label=row["parent_name"])
        ax.fill(angles, vals, alpha=0.08)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.35,1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR,"prompt_radar.png"), dpi=180)
    plt.close()

def plot_dataset_bar():
    path = os.path.join(RESULT_DIR, "round_5_results.json")
    if not os.path.exists(path):
        return
    df = load_and_flatten(path)
    if df.empty or "rouge1" not in df.columns:
        return
    summary = df.groupby("dataset")[["rouge1", "rougel", "fre"]].mean().round(3)
    fig, ax1 = plt.subplots(figsize=(8,5))
    width = 0.25
    x = range(len(summary.index))
    ax1.bar([i - width for i in x], summary["rouge1"], width=width, label="ROUGE-1")
    ax1.bar(x, summary["rougel"], width=width, label="ROUGE-L")
    ax2 = ax1.twinx()
    ax2.bar([i + width for i in x], summary["fre"], width=width, color="green", label="FRE")
    ax1.set_ylim(0,1)
    ax2.set_ylim(0,100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary.index, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR, "dataset_bar.png"), dpi=300)
    plt.close() 

def plot_baseline_vs_final():
    baseline_path=os.path.join(RESULT_DIR,"baseline.json")
    final_path=os.path.join(RESULT_DIR,"round_5_results.json")
    meet_path=os.path.join(RESULT_DIR,"meet_threshold.json")
    if not os.path.exists(baseline_path):
        return
    final_data=[]
    if os.path.exists(final_path):
        final_data.extend(json.load(open(final_path,"r",encoding="utf-8")))
    if os.path.exists(meet_path):
        for line in open(meet_path,"r",encoding="utf-8"):
            if line.strip():
                final_data.append(json.loads(line))
    if not final_data:
        return
    df_base=load_and_flatten(baseline_path)
    df_final=pd.DataFrame(final_data)
    if df_base.empty or df_final.empty or "rouge1" not in df_final.columns:
        return
    base_avg=df_base[["rouge1","rougel","fre"]].mean().round(3)
    final_avg=df_final[["rouge1","rougel","fre"]].mean().round(3)
    summary=pd.DataFrame({"ROUGE-1":[base_avg["rouge1"],final_avg["rouge1"]],"ROUGE-L":[base_avg["rougel"],final_avg["rougel"]],"FRE":[base_avg["fre"],final_avg["fre"]]},index=["Baseline","Final"])
    fig,ax1=plt.subplots(figsize=(8,5))
    ax1.plot(summary.index,summary["ROUGE-1"],marker="o",label="ROUGE-1")
    ax1.plot(summary.index,summary["ROUGE-L"],marker="s",label="ROUGE-L")
    ax2=ax1.twinx()
    ax2.plot(summary.index,summary["FRE"],marker="^",color="green",label="FRE")
    plt.tight_layout()
    plt.savefig(os.path.join(PIC_DIR,"baseline_vs_final.png"),dpi=300)
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
