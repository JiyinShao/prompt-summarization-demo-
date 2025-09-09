MODEL_NAME = "t5-base"
DATASETS = ["cnn", "xsum"]
SAMPLES_PER_DATASET = 5
SEED = 42

PROMPT_TEMPLATES = [
    "zero_shot.txt",
    "few_shot.txt",
    "instruction_based.txt",
    "pattern_based.txt",
    "target_audience.txt",
]

ACTIVE_MUTATIONS = [
    "none",
    "synonym_replacement",
    "prompt_rewriting",
    "style_instruction",
    "audience_information",
    "stepwise_prompt",
]

MAX_ROUNDS = 5
TOP_K = 5
THRESHOLDS = {"rouge1": 0.80, "fre": 80.0}

MUTATION_POOL = [
    "synonym_replacement",
    "prompt_rewriting",
    "style_instruction",
    "audience_information",
    "stepwise_prompt",
]

RESULT_DIR = "results"
