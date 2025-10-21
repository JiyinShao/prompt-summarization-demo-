# Prompt Summarization Evolution

This repository is part of my **Research Project 7100A** at the **University of Adelaide**.  
It investigates how different **prompt engineering strategies** and **mutation-based optimization methods** influence the quality of text summarization using **T5-based large language models (LLMs)**.

## Project Structure

prompt-summarization/<br>
│<br>
├── run_all.py # Automatically run all files<br>
├── generate_prompts.py # Generate prompts<br>
├── evolution.py # Implements prompt evolution across multiple rounds<br>
├── mutations.py # Defines mutation strategies for prompts<br>
├── baseline_generate.py # Runs baseline (no prompt/mutation) summarization<br>
├── visualize_results.py # Visualization of results<br>
├── evaluation.py # Calculates ROUGE-1, ROUGE-L, FRE, compression<br>
├── llm_utils.py # Model wrapper (T5 query / decoding utilities)<br>
├── data_utils.py # Helper for loading and preprocessing datasets<br>
├── sample_extraction.py # Selects articles from CNN and XSum datasets<br>
│<br>
├── data/<br>
│ ├── cnn_input.json # CNN/DailyMail test samples<br>
│ ├── xsum_input.json # XSum test samples<br>
│ ├── initial_prompts.json # Initial input prompts<br>
│ └── round_*_prompts.json # Input prompts in round n<br>
│<br>
├── results/ # All experimental outputs and figures<br>
│ ├── round_*_results.json # Per-round summarization results<br>
│ ├── meet_threshold.json # Prompts meeting threshold across rounds<br>
│ ├── baseline.json # Baseline summarization results<br>
│ └── pic/ # Other visualization charts<br>
│<br>
├── requirements.txt # Dependencies<br>
└── README.md<br>

## Current Progress
### 1. Prompting Strategies
Five styles are compared:
- zero_shot
- few_shot
- instruction_based
- pattern_based
- target_audience

### 2. Datasets
- **CNN/DailyMail** — long-form news articles  
- **XSUM** — short single-sentence summaries  

### 3. Mutation Methods
Each prompt undergoes up to five mutation types:
- Synonym replacement  
- Prompt rewriting  
- Style instruction  
- Audience information  
- Stepwise (multi-step) prompts  

### 4. Evaluation Metrics
Each generated summary is automatically evaluated using:
- **ROUGE-1**  
- **ROUGE-L**  
- **Flesch Reading Ease (FRE)**  
- **Compression ratio**

### 5. Baseline vs Evolution Comparison
- `baseline_generate.py`: Runs summarization without any prompt/mutation  
- `visualize_results.py`: Compares baseline vs final evolved prompts (bar/line charts)

### Setup

1. **Clone the repository**
   git clone https://github.com/JiyinShao/prompt-summarization-demo-.git
   cd prompt-summarization-demo-
2. **Install dependencies**
   pip install -r requirements.txt
3. **Extract articles from the database**
   python sample_extraction.py
4. **Run the summarization and evaluation pipeline**
   python evolution.py
5. **Aggregate results & generate charts**
   python visualize_results.py

This will:
 - Load test samples from each dataset
 - Apply all five prompt templates
 - Generate summaries using T5
 - Score outputs with ROUGE-1 / ROUGE-L / FRE / Compression Rate
 - Save results under results/
 - Export comparison charts 

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
