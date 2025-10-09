import evaluate
import textstat

rouge = evaluate.load("rouge")

def evaluate_summary(summary, reference, article):
    scores = rouge.compute(predictions=[summary], references=[reference])
    fre_score = textstat.flesch_reading_ease(summary)
    compression = len(summary.split()) / max(1, len(article.split()))
    return {
        "rouge1": scores["rouge1"],
        "rougel": scores["rougeL"],
        "fre": fre_score,
        "compression": compression
    }
