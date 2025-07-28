from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(inputs["input_ids"], max_length=64, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
<<<<<<< HEAD
    return output
=======
    return output
>>>>>>> 524bcf1 (Initial commit)
