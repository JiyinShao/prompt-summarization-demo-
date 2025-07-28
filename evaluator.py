import textstat
from rouge_score import rouge_scorer

def evaluate_fre(text):
    return textstat.flesch_reading_ease(text)

def evaluate_rouge(output, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, output)
    
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4)
    }
