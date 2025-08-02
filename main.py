from prompt_generator import generate_prompt
from data_loader import load_dataset
from llm_interface import query_llm
from evaluator import evaluate_fre, evaluate_rouge
import json
import os

datasets = ["cnn", "xsum"]
prompt_templates = [
    "zero_shot.txt",
    "few_shot.txt",
    "instruction_based.txt",
    "pattern_based.txt",
    "target_audience.txt"
]

os.makedirs("results", exist_ok=True)

for dataset_name in datasets:
    samples = load_dataset(dataset_name)

    results = []

    for template_name in prompt_templates:
        for sample in samples:
            prompt = generate_prompt(sample["article"], template_file=template_name)
            output = query_llm(prompt)

            fre = evaluate_fre(output)
            rouge = evaluate_rouge(output, sample["reference"])

            results.append({
                "id": sample["id"],
                "dataset": dataset_name,
                "template": template_name,
                "output": output,
                "fre": fre,
                "rouge1": rouge["rouge1"],
                "rougeL": rouge["rougeL"]
            })

    result_file = f"results/{dataset_name}_prompt_eval.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[✓] Saved results to {result_file}")
