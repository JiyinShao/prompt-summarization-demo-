from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base") 
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def query_llm(prompt):
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).input_ids

    output_ids = model.generate(
    input_ids,
    num_beams=4,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    min_length=40,
    max_length=120,
    early_stopping=True
)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
