# utils.py
import torch

def collate_fn(batch):
    images, reports = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(reports)

def compute_bleu(preds, refs):
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except ImportError:
        raise ImportError("Please install sacrebleu to compute BLEU scores.")

def compute_rouge(preds, refs):
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(r, p) for p, r in zip(preds, refs)]
        avg_scores = {
            'rouge1': sum([s['rouge1'].fmeasure for s in scores]) / len(scores),
            'rougeL': sum([s['rougeL'].fmeasure for s in scores]) / len(scores)
        }
        return avg_scores
    except ImportError:
        raise ImportError("Please install rouge_score to compute ROUGE scores.")
