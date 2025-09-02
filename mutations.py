import random
import re

def _append_constraints(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Constraints:\n"
        "- Output 3–5 sentences (≤ 60 words total).\n"
        "- Be factual and concise; no preface or headings.\n"
        "- Reuse key nouns/verbs from the source when possible.\n"
        "- Keep names, numbers, places, and dates unchanged.\n"
        "- Do not invent details; do not quote verbatim.\n"
    )

def synonym_replacement(prompt: str) -> str:
    vocab = {
        "clearly": ["plainly", "explicitly"],
        "brief": ["concise", "short"],
        "important": ["key", "crucial"],
        "main": ["primary", "central"],
        "points": ["aspects", "facts"],
        "explain": ["describe", "clarify"],
    }
    tokens = re.split(r"(\W+)", prompt)
    out = []
    for tok in tokens:
        low = tok.lower()
        if low in {"summary","summarize","article","document","text","passage"}:
            out.append(tok)
            continue
        if low in vocab and random.random() < 0.12:
            cand = random.choice(vocab[low])
            cand = cand.capitalize() if tok[:1].isupper() else cand
            out.append(cand)
        else:
            out.append(tok)
    base = "".join(out)
    return _append_constraints(base)

def prompt_rewriting(prompt: str) -> str:
    p = prompt.strip()
    p = re.sub(r"^\s*summarize\s*[:\-]?", "Write a brief, factual summary:", p, flags=re.I)
    p = p.replace("Summarize the following", "Write a brief, factual summary of the following")
    rewritten = (
        f"{p}\n\n"
        "Instruction:\n"
        "- Write a coherent summary in 3–5 sentences.\n"
        "- Prioritize the who/what/where/when/why.\n"
        "- Prefer wording that appears in the source to maximize fidelity.\n"
        "- Avoid bullets and formatting; output only the final summary."
    )
    return rewritten

def style_instruction(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Style:\n"
        "- Clear, journalistic tone.\n"
        "- 3–5 sentences; ≤ 35 words total.\n"
        "- Use active voice and concrete subjects.\n"
        "- Reuse key phrases from the article to maintain alignment.\n"
        "- No lists, no preambles, no quotations."
    )

def audience_information(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Audience:\n"
        "- Non-expert adult reader.\n"
        "- Keep terminology accurate; do not oversimplify named entities.\n"
        "- Explain briefly only if a term is uncommon; otherwise reuse the original wording.\n"
        "- Output 3–5 concise sentences without headings."
    )

def stepwise_prompt(prompt: str) -> str:
    return (
        f"{prompt}\n\n"
        "Process:\n"
        "1) Identify 3–5 essential facts (subjects, actions, outcomes, numbers, places).\n"
        "2) Compress them into 2–3 sentences while preserving names and numbers.\n"
        "3) Prefer wording from the source when possible to ensure fidelity.\n"
        "Output only the final summary, no list, no preface."
    )

MUTATIONS = {
    "none": lambda p: _append_constraints(p),
    "synonym_replacement": synonym_replacement,
    "prompt_rewriting": prompt_rewriting,
    "style_instruction": style_instruction,
    "audience_information": audience_information,
    "stepwise_prompt": stepwise_prompt,
}
