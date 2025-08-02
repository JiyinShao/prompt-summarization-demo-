def generate_prompt(text, template_file="zero_shot.txt"):
    with open(f"prompts/prompt_templates/{template_file}", "r", encoding="utf-8") as f:
        template = f.read()
    return template.replace("{text}", text)
