# Prompt Summarization Demo

This repository is part of my **Research Project 7100A** at the **University of Adelaide**. It explores how different **prompt engineering strategies** influence the quality of text summarization using **large language models (LLMs)** such as T5.

## Project Structure

prompt-summarization-demo-/<br>
│<br>
├── main.py # Run the end-to-end prompt → LLM → evaluation pipeline<br>
├── prompt_generator.py # Builds prompts from template files<br>
├── llm_interface.py # T5 model wrapper (tokenize → generate → decode)<br>
├── evaluator.py # ROUGE-1 / ROUGE-L / Flesch Reading Ease (FRE)<br>
├── analyze_results.py # Aggregation & visualization (CSV + charts)<br>
│<br>
├── prompts/<br>
│ ├── zero_shot.txt<br>
│ ├── few_shot.txt<br>
│ ├── instruction_based.txt<br>
│ ├── pattern_based.txt<br>
│ └── target_audience.txt # Five prompt strategies used in current experiments<br>
│<br>
├── results/<br>
│ ├── cnn_prompt_eval_5.json # Per-sample scores for CNN/DailyMail<br>
│ ├── xsum_prompt_eval_5.json # Per-sample scores for XSum<br>
│ ├── all_results.csv # Merged results (created by analyze_results.py)<br>
│ ├── mean_by_template.csv # Mean scores by prompt template<br>
│ ├── mean_by_template_mutation.csv (if mutation is added later)<br>
│ ├── rouge1_by_template_dataset.png<br>
│ ├── rougeL_by_template_dataset.png<br>
│ └── fre_by_template_dataset.png # Summary charts (created by analyze_results.py)<br>
│<br>
├── requirements.txt # Python dependencies<br>
└── README.md<br>

## Current Progress

- **Five prompting strategies**
  - zero_shot.txt, few_shot.txt, instruction_based.txt, pattern_based.txt, target_audience.txt
- **Two datasets**
  - CNN/DailyMail (cnn_dailymail, split: test)
  - XSum (EdinburghNLP/xsum, split: test, parquet revision)
- **Automatic evaluation**
  - **ROUGE-1**, **ROUGE-L**, **Flesch Reading Ease (FRE)**
- **Batch execution**
  - By default: 5 articles × 5 prompts × 2 datasets = 50 evaluations
- **Result aggregation & visualization**
  - Merged CSV + bar charts comparing templates across datasets

### Setup

1. **Clone the repository**
   git clone https://github.com/JiyinShao/prompt-summarization-demo-.git
   cd prompt-summarization-demo-
2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the summarization and evaluation pipeline**
   python main.py

This will:
 - Load 5 test samples from each dataset (configurable)
 - Apply all five prompt templates
 - Generate summaries using T5
 - Score outputs with ROUGE-1 / ROUGE-L / FRE
 - Save per-dataset results to:
 - results/cnn_prompt_eval_5.json
 - results/xsum_prompt_eval_5.json

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
