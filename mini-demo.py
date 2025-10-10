import os
import json
from llm_utils import query_t5
from evaluation import evaluate_summary

DATA_DIR = "data"

def load_sample():
    with open(os.path.join(DATA_DIR, "cnn_input.json"), "r", encoding="utf-8") as f:
        cnn_data = json.load(f)
    with open(os.path.join(DATA_DIR, "xsum_input.json"), "r", encoding="utf-8") as f:
        xsum_data = json.load(f)
    cnn_article, cnn_ref = cnn_data[0]["article"], cnn_data[0]["reference"]
    xsum_article, xsum_ref = xsum_data[0]["article"], xsum_data[0]["reference"]
    return ("cnn", cnn_article, cnn_ref), ("xsum", xsum_article, xsum_ref)

def summarize_with_tuned_model(article, prompt):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    MODEL_NAME = "mrm8488/t5-base-finetuned-summarize-news"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    input_text = f"{prompt} {article}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def evaluate_prompt(prompt, article, reference):
    summary = summarize_with_tuned_model(article, prompt)
    scores = evaluate_summary(summary, reference, article)
    return summary, scores

def main():
    cnn_sample, xsum_sample = load_sample()
    prompt1 = "summarize:"
    prompt2 = "Can you list the three most important facts for daily news readers from this article?"

    print("\n=== Demo Showcase (Fine-tuned Summarization Model) ===")

    for dataset_name, article, reference in [cnn_sample, xsum_sample]:
        print(f"\n--- {dataset_name.upper()} SAMPLE ---")
        for i, prompt in enumerate([prompt1, prompt2], start=1):
            summary, scores = evaluate_prompt(prompt, article, reference)
            print(f"\nPrompt {i}: {prompt}")
            print(f"ROUGE-1: {scores['rouge1']:.3f}")
            print(f"ROUGE-L: {scores['rougel']:.3f}")
            print(f"FRE: {scores['fre']:.2f}")
            print(f"Summary: {summary[:300]}...")

if __name__ == "__main__":
    main()
