from datasets import load_dataset
import random

def load_dataset_samples(dataset_name, num_samples=5):
    if dataset_name == "cnn":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
    elif dataset_name == "xsum":
        dataset = load_dataset("xsum", split="test")
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    samples = random.sample(list(dataset), num_samples)

    return [
        {
            "id": f"{dataset_name}_{i+1:03}",
            "article": s["article"],
            "reference": s["highlights"] if dataset_name == "cnn" else s["summary"]
        }
        for i, s in enumerate(samples)
    ]
