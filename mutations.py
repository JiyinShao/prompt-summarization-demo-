import random

def synonym_replacement(prompt: str) -> str:
    synonyms = {
        "summary": ["overview", "brief", "recap"],
        "summarize": ["condense", "shorten", "compress"],
        "article": ["text", "document", "passage"],
        "explain": ["describe", "clarify", "illustrate"],
        "important": ["crucial", "key", "essential"],
        "main": ["primary", "core", "principal"],
        "points": ["aspects", "items", "factors"],
        "highlight": ["emphasize", "underscore", "stress"],
    }
    words = prompt.split()
    new_words = []
    for w in words:
        bare = w.strip(".,!?;:()[]{}\"'")
        lw = bare.lower()
        if lw in synonyms and random.random() < 0.3:
            rep = random.choice(synonyms[lw])
            rep = rep.capitalize() if bare[:1].isupper() else rep
            new = w.replace(bare, rep)
            new_words.append(new)
        else:
            new_words.append(w)
    return " ".join(new_words)

def prompt_rewriting(prompt: str) -> str:
    p = prompt.strip()
    lower = p.lower()
    if lower.startswith("summarize the following"):
        return p.replace("Summarize the following", "The following should be summarized")
    if lower.startswith("summarize:"):
        return p.replace("summarize:", "the following should be summarized:")
    return f"{p}\n\nRewritten: Ensure key points are highlighted and clauses are ordered for clarity."

def style_instruction(prompt: str) -> str:
    return f"{prompt}\n\nStyle: Use simple English, concise sentences, and bullet points if appropriate."

def audience_information(prompt: str) -> str:
    return f"{prompt}\n\nAudience: Write for a 10-year-old reader with clear, accessible language."

def stepwise_prompt(prompt: str) -> str:
    scaffold = (
        "\n\nProcess:\n"
        "1) Extract 3–5 key points from the text.\n"
        "2) Rephrase them in your own words.\n"
        "3) Compose a coherent final summary.\n"
        "Output only the final summary."
    )
    return prompt + scaffold

MUTATIONS = {
    "none": lambda p: p,
    "synonym_replacement": synonym_replacement,
    "prompt_rewriting": prompt_rewriting,
    "style_instruction": style_instruction,
    "audience_information": audience_information,
    "stepwise_prompt": stepwise_prompt,
}
