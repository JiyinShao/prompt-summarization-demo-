import random

def synonym_replacement(prompt: str) -> str:
    synonyms = {
        "summary": ["overview", "brief", "recap"],
        "summarize": ["condense", "shorten", "compress"],
        "article": ["text", "document", "passage"],
        "explain": ["describe", "clarify", "illustrate"],
        "important": ["crucial", "key", "essential"]
    }
    words = prompt.split()
    new_words = []
    for w in words:
        lw = w.lower().strip(".,!?")
        if lw in synonyms and random.random() < 0.3:
            new_words.append(random.choice(synonyms[lw]))
        else:
            new_words.append(w)
    return " ".join(new_words)

def prompt_rewriting(prompt: str) -> str:
    if prompt.strip().lower().startswith("summarize"):
        return prompt.replace("Summarize the following", 
                              "The following should be summarized")
    return f"{prompt}\n\nRewritten: Make sure to highlight main ideas clearly."

MUTATIONS = {
    "none": lambda p: p,
    "synonym_replacement": synonym_replacement,
    "prompt_rewriting": prompt_rewriting,
}
