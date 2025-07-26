# Prompt Summarization Demo

This repository is part of my Research Project 7100A at the University of Adelaide. It explores how different prompt engineering strategies affect summarization performance using large language models (LLMs).

## Project Structure

prompt-summarization-demo/<br>
│<br>
├── main.py # Entry point for running prompt → summary → evaluation<br>
├── config.py # Configuration file for thresholds, model settings, etc.<br>
├── prompt_generator.py # Loads and renders prompt templates<br>
├── llm_interface.py # Interface to T5 model for summary generation<br>
├── evaluator.py # FRE-based readability evaluation<br>
├── requirements.txt # Required Python packages<br>
│<br>
├── data/<br>
│ └── input_texts.json # Sample input from CNN dataset (article + reference)<br>
│<br>
├── results/<br>
│ └── final_outputs.json # Generated summaries + evaluation results<br>
│<br>
├── README.md # Project overview (this file)<br>

## Current Progress

- **Initial pipeline structure completed**  
  - Prompt → LLM (T5) → Output → FRE Evaluation → JSON Logging
  - Modular design allows for future integration of mutation strategies and ROUGE evaluation

- **Next Steps (Planned)**  
  - Implement multiple prompt mutation strategies  
  - Add ROUGE-based quality scoring  
  - Allow multi-turn mutation loops for underperforming outputs  
  - Build comparative visualization of different prompt strategies

## Usage

This repository contains both prompt design notebooks and a runnable pipeline for evaluating summarization quality using T5.

### Setup

1. **Clone the repository**
   git clone https://github.com/JiyinShao/prompt-summarization-demo.git
   cd prompt-summarization-demo
2. **Install dependencies**
   pip install -r requirements.txt
3. **Run the summarization and evaluation pipeline**
   python main.py

This will:
 - Load CNN-style news articles from data/input_texts.json
 - Generate summaries using T5 and your selected prompt style
 - Evaluate outputs using the Flesch Reading Ease (FRE) score
 - Save all outputs and scores to results/final_outputs.json

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
