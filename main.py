from data_utils import load_datasets
from evolution import run_evolution

INITIAL_PROMPTS = [
    {
        "name": "zero_shot",
        "text": "Can you give me a short summary of this article?"
    },
    {
        "name": "few_shot",
        "text": """Here are some examples of articles and their summaries:

        Example 1:
        Article: A storm hit northern France yesterday, damaging homes and cutting power.
        Summary: A storm in northern France caused damage and power outages.

        Example 2:
        Article: A new study shows coffee may help reduce the risk of heart disease.
        Summary: Researchers found coffee drinkers had lower heart disease risk.

        Now here is a new article. Please write its summary:"""
    },
    {
        "name": "instruction_based",
        "text": "Please explain the main points of this article briefly in three sentences: "
    },
    {
        "name": "pattern_based",
        "text": "Can you list the three most important facts from this article as bullet points?"
    },
    {
        "name": "target_audience",
        "text": "Can you rewrite this article so that it’s easy to understand for everyday readers?"
    }
]

DATASETS = ["cnn_dailymail", "EdinburghNLP/xsum"]

if __name__ == "__main__":
    datasets = load_datasets(DATASETS, n_samples=2)
    run_evolution(datasets, INITIAL_PROMPTS, max_rounds=5, top_k=5)
