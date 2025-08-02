# Prompt Summarization Demo

This repository is part of my **Research Project 7100A** at the **University of Adelaide**. It explores how different **prompt engineering strategies** influence the quality of text summarization using **large language models (LLMs)** such as T5.

## Project Structure

prompt-summarization-demo-/<br>
│<br>
├── main.py # Entry point for running full prompt→LLM→eval pipeline<br>
├── config.py # Configurable settings (e.g., model, prompt, dataset)<br>
├── data_loader.py # Data loading utilities (CNN, XSUM, etc.)<br>
├── llm_interface.py # T5 model wrapper for text generation<br>
├── evaluator.py # Evaluation: ROUGE and FRE scoring<br>
│<br>
├── data/<br>
│ ├── input_texts.json # CNN input samples<br>
│ └── xsum_sample.json # XSUM input samples<br>
│<br>
├── prompts/<br>
│ └── prompt_templates/<br>
│ ├── zero_shot.txt<br>
│ ├── few_shot.txt<br>
│ ├── instruction_based.txt<br>
│ ├── pattern_based.txt<br>
│ └── target_audience.txt # 5 prompt strategies used in current experiment<br>
│<br>
├── results/<br>
│ ├── cnn_prompt_eval.json # Evaluation output for CNN dataset<br>
│ └── xsum_prompt_eval.json # Evaluation output for XSUM dataset<br>
│<br>
├── requirements.txt # Python package dependencies<br>
├── README.md<br>

## Current Progress

- Prompt-to-summary pipeline using **T5-base**
- Supports **five prompting strategies**:
  - zero_shot.txt
  - few_shot.txt
  - instruction_based.txt
  - pattern_based.txt
  - target_audience.txt
- Evaluation using:
  - **ROUGE-1**
  - **ROUGE-L**
  - **Flesch Reading Ease**
- Automatic output saving in structured JSON format

### Setup

1. **Clone the repository**
   git clone https://github.com/JiyinShao/prompt-summarization-demo-.git
   cd prompt-summarization-demo-
2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the summarization and evaluation pipeline**
   python main.py

This will:
 - Load selected dataset (data/input_texts.json or xsum_sample.json)
 - Apply all five prompt strategies
 - Generate summaries using T5-base model
 - Evaluate outputs using the ROUGE and FRE score
 - Save all outputs and scores to results/cnn_prompt_eval.json and results/xsum_prompt_eval.json

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
