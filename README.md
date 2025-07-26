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

## Current Progress

- **Baseline prompt strategies implemented**  
  Designed and tested 5 core prompt styles:
  - Zero-shot  
  - Few-shot  
  - Instruction-based  
  - Pattern-based  
  - Target audience–oriented

- **Initial pipeline structure completed**  
  - Prompt → LLM (T5) → Output → FRE Evaluation → JSON Logging
  - Modular design allows for future integration of mutation strategies and ROUGE evaluation

- **Next Steps (Planned)**  
  - Implement multiple prompt mutation strategies  
  - Add ROUGE-based quality scoring  
  - Allow multi-turn mutation loops for underperforming outputs  
  - Build comparative visualization of different prompt strategies


## Usage

Right now this repo is only for storing prompt designs. Notebook and evaluation will be uploaded soon.

## Contact

Author: **Jiyin Shao**  
Email: [a1903968@adelaide.edu.au]  
University of Adelaide, Research Project 7100A
