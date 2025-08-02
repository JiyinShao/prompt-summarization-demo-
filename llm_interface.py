from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

def query_llm(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_length=100)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output
