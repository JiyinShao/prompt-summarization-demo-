import os
import json
from generate_prompts import generate_prompt_combinations
from evolution import run_evolution

DATA_DIR = "data"
RESULT_DIR = "results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MAX_ROUNDS = 5
TOP_K = 5
ROUGE1_THRESHOLD = 0.5

def run_all_rounds():
    round_idx = 1
    input_file = os.path.join(DATA_DIR, "initial_prompts.json")

    with open(os.path.join(DATA_DIR, "cnn_input.json"), "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open(os.path.join(DATA_DIR, "xsum_input.json"), "r", encoding="utf-8") as f:
        xsum_data = json.load(f)

    datasets = {
        "cnn": [{"id": i["id"], "article": i["article"], "reference": i["reference"]} for i in cnn_data],
        "xsum": [{"id": i["id"], "article": i["article"], "reference": i["reference"]} for i in xsum_data]
    }

    while round_idx <= MAX_ROUNDS:
        print(f"\n=== Round {round_idx} started ===")
        output_prompt_file = os.path.join(RESULT_DIR, f"round_{round_idx}_prompts.json")

        generate_prompt_combinations(input_file, output_prompt_file)
        run_evolution(datasets, output_prompt_file, round_idx=round_idx, rouge1_threshold=ROUGE1_THRESHOLD, top_k=TOP_K)

        next_round_prompts = os.path.join(DATA_DIR, f"round_{round_idx+1}_prompts.json")
        if not os.path.exists(next_round_prompts):
            print(f"Evolution finished early at round {round_idx}")
            break

        input_file = next_round_prompts
        round_idx += 1

    print("\nAll rounds completed.")
    print(f"Results saved in: {RESULT_DIR}")

if __name__ == "__main__":
    run_all_rounds()
