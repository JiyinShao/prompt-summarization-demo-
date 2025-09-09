import random
import re

def synonym_replacement(prompt: str) -> str:
    vocab = {
        "summary": ["brief summary", "concise summary"],
        "summarize": ["write a summary of", "produce a summary of"],
        "article": ["news article", "source article"],
        "key": ["important", "crucial"],
    }
    tokens = re.split(r"(\W+)", prompt)
    out = []
    for tok in tokens:
        lw = tok.lower()
        if lw in vocab and random.random() < 0.1:
            cand = random.choice(vocab[lw])
            out.append(cand.capitalize() if tok[:1].isupper() else cand)
        else:
            out.append(tok)
    return "".join(out) + "\n\nHint: Reuse key phrases and named entities from the article to preserve fidelity."

def prompt_rewriting(prompt: str) -> str:
    p = re.sub(r"^\s*summarize\s*[:\-]?", "Write a clear and accurate summary:", prompt, flags=re.I)
    p = p.replace("Summarize the following", "Write a clear and accurate summary of the following")
    return (
        f"{p}\n\n"
        "Instruction:\n"
        "- Focus on who, what, where, when, and outcomes.\n"
        "- Prefer wording that appears in the article; avoid paraphrasing when not necessary.\n"
        "- Output only the final summary."
    )

def style_instruction(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Style:\n"
        "- Natural news tone, direct and informative.\n"
        "- Keep sentences compact but complete; avoid lists and meta commentary.\n"
        "- Use concrete nouns and active verbs; reuse source wording when appropriate."
    )

def audience_information(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Audience:\n"
        "- General news readers; keep terminology accurate and names unchanged.\n"
    )

def stepwise_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Process:\n"
        "1) Read the article. \n "
        "2) Identify key facts: actors, actions, outcomes, numbers, places.\n"
        "3) Compose a fluent summary that integrates these facts.\n"
        "4) Output only the final summary paragraph (no steps or notes)."
    )

MUTATIONS = {
    "none": lambda p: p,
    "synonym_replacement": synonym_replacement,
    "prompt_rewriting": prompt_rewriting,
    "style_instruction": style_instruction,
    "audience_information": audience_information,
    "stepwise_prompt": stepwise_prompt,
}
