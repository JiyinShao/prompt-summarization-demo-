import os

TEMPLATE_DIR = os.path.join("prompts", "prompt_templates")

def generate_prompt(article: str, template_file: str) -> str:
    path = os.path.join(TEMPLATE_DIR, template_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        template = f.read()

    if "{text}" in template:
        return template.replace("{text}", article)
    else:
        return f"{template}\n\nArticle:\n{article}"
