from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def query_t5(input_text: str, max_length=128) -> str:
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        min_length=30,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def mutate_with_llm(base_prompt: str, style: str) -> list[str]:
    meta_prompt = f"Generate new prompts based on the following prompt with the instruction: {style}\nOriginal prompt: {base_prompt}"
    new_prompt = query_t5(meta_prompt, max_length=64)
    return [new_prompt]

def summarize_with_prompt(article: str, prompt: str) -> str:
    input_text = f"{prompt}\n\nArticle: {article}"
    return query_t5(input_text, max_length=128)
