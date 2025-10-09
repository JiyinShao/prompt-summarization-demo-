from llm_utils import mutate_with_llm

MUTATION_GUIDELINES = {
    "synonym_replacement": "Rewrite the prompt by replacing some keywords with synonyms while keeping the meaning.",
    "prompt_rewriting": "Rephrase the prompt to make it clearer and more concise, but keep the same intent.",
    "style_instruction": "Add explicit instructions about the style of the expected summary, e.g. bullet points, step-by-step, or three-sentence output.",
    "audience_information": "Modify the prompt so that the summary is tailored to a specific audience, such as children, students, or experts.",
    "stepwise_prompt": "Always append a step-by-step instruction template: 1) Read the article, 2) Extract key facts, 3) Write a concise summary."
}

def apply_stepwise(prompt: str) -> str:
    return f"{MUTATION_GUIDELINES['stepwise_prompt']}\nOriginal prompt: {prompt}"

def mutate(prompt: str, mtype: str) -> list[str]:
    if mtype == "stepwise_prompt":
        return [apply_stepwise(prompt)]
    else:
        return mutate_with_llm(prompt, style=MUTATION_GUIDELINES[mtype], n=2)
