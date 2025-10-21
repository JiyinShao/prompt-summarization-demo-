import json
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from mutations import MUTATION_GUIDELINES

PARAPHRASE_MODEL = "google/flan-t5-large"

tokenizer = T5Tokenizer.from_pretrained(PARAPHRASE_MODEL)
model = T5ForConditionalGeneration.from_pretrained(
    PARAPHRASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

MUTATION_NAMES = list(MUTATION_GUIDELINES.keys())

def generate_with_model(base_prompt: str, instruction: str) -> str:
    input_text = f"""
You are a prompt rewriting assistant.
Follow the given instruction to modify the following prompt.

Instruction: {instruction}
Prompt: {base_prompt}

Return only the rewritten prompt:
"""
    encoding = tokenizer.encode_plus(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    outputs = model.generate(
        **encoding,
        max_length=96,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
        num_return_sequences=1,
        repetition_penalty=1.15,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    for marker in ["Instruction:", "Prompt:", "Return only"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    return text

def generate_new_prompt(base_prompt: str, mutation_type: str) -> str:
    if mutation_type == "stepwise_prompt":
        stepwise_suffix = (
            " Always follow these steps: 1) Read the article, 2) Extract key facts, 3) Write a concise summary."
        )
        return f"{base_prompt.strip()}{stepwise_suffix}"
    else:
        instruction = MUTATION_GUIDELINES[mutation_type]
        return generate_with_model(base_prompt, instruction)

def generate_prompt_combinations(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        initial_prompts = json.load(f)

    total = len(initial_prompts) * (1 + len(MUTATION_NAMES))
    count = 0
    all_candidates = []

    for p in initial_prompts:
        parent_name = p["name"]
        parent_text = p["text"]

        all_candidates.append({
            "parent_name": parent_name,
            "mutation": "none",
            "prompt_text": parent_text
        })
        count += 1
        print(f"[{count}/{total}] generated")

        for mtype in MUTATION_NAMES:
            new_prompt = generate_new_prompt(parent_text, mtype)
            all_candidates.append({
                "parent_name": parent_name,
                "mutation": mtype,
                "prompt_text": new_prompt
            })
            count += 1
            print(f"[{count}/{total}] generated")

    os.makedirs("results", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, ensure_ascii=False, indent=2)

    print(f"\nAll {len(all_candidates)} prompt candidates saved to {output_file}")

if __name__ == "__main__":
    generate_prompt_combinations("data/initial_prompts.json", "results/round_1_prompts.json")
