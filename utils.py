import torch

def collate_fn(batch):
    """
    Collate function to batch images and reports.
    Args:
        batch (list of tuples): Each element is (image_tensor, report_str)
    Returns:
        images (Tensor): Batched images of shape (B, C, H, W)
        reports (list of str): List of report strings
    """
    images, reports = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(reports)


def compute_bleu(preds, refs):
    """
    Compute BLEU score using sacrebleu.
    Args:
        preds (list of str): Generated texts
        refs (list of str): Reference texts
    Returns:
        float: BLEU score
    """
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(preds, [refs]).score
    except ImportError:
        raise ImportError("Please install sacrebleu to compute BLEU scores.")


def compute_rouge(preds, refs):
    """
    Compute average ROUGE-1 and ROUGE-L F1 scores.
    Args:
        preds (list of str)
        refs (list of str)
    Returns:
        dict: {'rouge1': float, 'rougeL': float}
    """
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