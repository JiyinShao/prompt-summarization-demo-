import json
from prompt_generator import generate_prompt
from llm_interface import query_llm
from evaluator import evaluate_fre
from config import FRE_THRESHOLD

# 1. Load input data
with open("data/input_texts.json", "r", encoding="utf-8") as f:
    samples = json.load(f)

results = []

# 2. Process each article
for sample in samples:
    article = sample["article"]
    reference = sample["reference"]
    prompt = generate_prompt(article)
    output = query_llm(prompt)
    fre_score = evaluate_fre(output)

    print(f"Prompt:\n{prompt}\n")
    print(f"Output:\n{output}\n")
    print(f"FRE Score: {fre_score:.2f}\n{'='*50}\n")

    results.append({
        "id": sample["id"],
        "prompt": prompt,
        "output": output,
        "fre_score": fre_score,
        "reference": reference
    })

# 3. Save results
with open("results/final_outputs.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
