# Prompt Summarization Demo

This repository is part of my **Research Project 7100A** at the **University of Adelaide**. It explores how different **prompt engineering strategies** influence the quality of text summarization using **large language models (LLMs)** such as T5.

## Project Structure

prompt-summarization-demo-/ <br>
│<br>
├── main.py                  # Run the first loop of optimizer<br>
├── evolution.py             # Select the TOP 5 results and continue generation<br>
├── prompt_generator.py      # Builds prompts from template files<br>
├── llm_interface.py         # T5 model wrapper (tokenize → generate → decode)<br>
├── evaluator.py             # ROUGE-1 / ROUGE-L / Flesch Reading Ease (FRE)<br>
├── analyze_results.py       # Aggregation & visualization (CSV + charts)<br>
├── mutations.py             # Mutation strategies for prompts (lexical, structural, style, audience, stepwise)<br>
├── config.py                # Define the Threshold of score and MAX rounds<br>
├── sample_extraction.py     # Extract samples from the database<br>
│<br>
├── prompts/<br>
│   ├── zero_shot.txt<br>
│   ├── few_shot.txt<br>
│   ├── instruction_based.txt<br>
│   ├── pattern_based.txt<br>
│   └── target_audience.txt   # Five prompt strategies used in current experiments<br>
│<br>
├── results/                 # Evaluation outputs (see details below)<br>
│<br>
├── requirements.txt         # Python dependencies<br>
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
  - First Round: 2 datasets x 5 articles × 5 prompts x 6 mutations = 300 evaluations
- **Mutation strategies**
  - None
  - Synonym replacement
  - Prompt rewriting
  - Add style instruction
  - Add audience information
  - Stepwise prompt
- **Result aggregation & visualization**
  - Merged CSV + bar charts comparing templates across datasets

### Results Directory
The results/ folder contains three types of output:

## 1. Raw per-sample results (JSON)
Store individual evaluation results for each round. Store the selected combination in each round. 

## 2. Visualizations (PNG)
Allow direct comparison of prompting strategies and mutation effects.

### Setup

1. **Clone the repository**
   git clone https://github.com/JiyinShao/prompt-summarization-demo-.git
   cd prompt-summarization-demo-
2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the summarization and evaluation pipeline**
   python main.py
4. **Aggregate results & generate charts**
   python analyze_results.py

This will:
 - Load 2 test samples from each dataset (configurable)
 - Apply all five prompt templates
 - Generate summaries using T5
 - Score outputs with ROUGE-1 / ROUGE-L / FRE
 - Save results under results/
 - Export comparison charts 

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
