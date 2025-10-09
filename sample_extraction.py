import os, json, random
from datasets import load_dataset

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def sample_and_save(dataset_name, split, k, out_path, article_key, summary_key):
    print(f"[loading] {dataset_name}")
    ds = load_dataset(dataset_name, split=split)

    items = random.sample(list(ds), k)

    records = []
    for i, s in enumerate(items, start=1):
        records.append({
            "id": f"{dataset_name}_{i:05}",
            "article": s[article_key],
            "reference": s[summary_key]
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_path} ({len(records)} samples)")

def main():
    sample_and_save(
        dataset_name="cnn_dailymail",
        split="test",
        k=20,
        out_path=os.path.join(OUT_DIR, "cnn_input.json"),
        article_key="article",
        summary_key="highlights"
    )

    sample_and_save(
        dataset_name="EdinburghNLP/xsum",
        split="test",
        k=20,
        out_path=os.path.join(OUT_DIR, "xsum_input.json"),
        article_key="document",
        summary_key="summary"
    )

if __name__ == "__main__":
    main()
