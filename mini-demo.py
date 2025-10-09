import os
import json
from llm_utils import summarize_with_prompt
from evaluation import evaluate_summary

DATA_DIR="data"

def load_sample():
    with open(os.path.join(DATA_DIR,"cnn_input.json"),"r",encoding="utf-8") as f:
        cnn_data=json.load(f)
    with open(os.path.join(DATA_DIR,"xsum_input.json"),"r",encoding="utf-8") as f:
        xsum_data=json.load(f)
    cnn_article,cnn_ref=cnn_data[0]["article"],cnn_data[0]["reference"]
    xsum_article,xsum_ref=xsum_data[0]["article"],xsum_data[0]["reference"]
    return ("cnn",cnn_article,cnn_ref),("xsum",xsum_article,xsum_ref)

def evaluate_prompt(prompt,article,reference):
    summary=summarize_with_prompt(article,prompt)
    scores=evaluate_summary(summary,reference,article)
    return summary,scores

def main():
    cnn_sample,xsum_sample=load_sample()
    print("\n=== Demo Showcase ===")
    prompt1=input("Enter Prompt 1: \n")
    prompt2=input("Enter Prompt 2: \n")

    for dataset_name,article,reference in [cnn_sample,xsum_sample]:
        print(f"\n--- {dataset_name.upper()} SAMPLE ---")
        for i,prompt in enumerate([prompt1,prompt2],start=1):
            summary,scores=evaluate_prompt(prompt,article,reference)
            print(f"\nPrompt {i}: {prompt}")
            print(f"ROUGE-1: {scores['rouge1']:.3f}")
            print(f"ROUGE-L: {scores['rougel']:.3f}")
            print(f"FRE: {scores['fre']:.2f}")
            print(f"Summary: {summary[:300]}...")

if __name__=="__main__":
    main()
