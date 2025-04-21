# evaluation.py
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from config import Config  # ADD THIS LINE to import Config
import nltk

nltk.download('wordnet')
nltk.download('punkt')

def evaluate_model(model, data_loader, device, config=Config):
    model.eval()
    total_loss = 0
    gens, refs = [], []

    with torch.no_grad():
        for images, reports in data_loader:
            images = images.to(device)

            input_ids, attention_mask = model.text_decoder.encode_text(
                reports, max_length=config.max_len
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(images, reports, labels=input_ids)
            total_loss += outputs.loss.item()

            # Skip empty inputs
            if input_ids.size(1) == 0:
                continue

            # Fix bos_token_id issue
            tokenizer = model.text_decoder.tokenizer
            bos_token_id = getattr(tokenizer, 'bos_token_id', None)
            if bos_token_id is None:
                bos_token_id = getattr(tokenizer, 'cls_token_id', None)

            generated_ids = model.text_decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bos_token_id=bos_token_id
            )
            for gen, ref in zip(generated_ids, reports):
                gens.append(gen)
                refs.append(ref)

    avg_loss = total_loss / len(data_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    # BLEU Metrics Calculation
    smooth = SmoothingFunction().method1
    gen_tokens = [word_tokenize(g.lower()) for g in gens]
    ref_tokens = [[word_tokenize(r.lower())] for r in refs]
    
    bleu1 = corpus_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(ref_tokens, gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(ref_tokens, gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    # METEOR
    m_scores = [meteor_score([word_tokenize(r)], word_tokenize(g)) for r, g in zip(refs, gens)]
    meteor = sum(m_scores) / len(m_scores)

    # ROUGE-L
    def lcs(x, y):
        m, n = len(x), len(y)
        L = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                L[i][j] = L[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(L[i - 1][j], L[i][j - 1])
        return L[m][n]
    
    scores = []
    for r, g in zip(refs, gens):
        rt, gt = word_tokenize(r), word_tokenize(g)
        l = lcs(rt, gt)
        rec = l / len(rt) if rt else 0
        prec = l / len(gt) if gt else 0
        scores.append((2 * rec * prec / (rec + prec)) if rec + prec else 0)
    
    rougeL = sum(scores) / len(scores)

    # Return metrics
    return {
        "loss": avg_loss,
        "bleu1": bleu1,
        "bleu2": bleu2,
        "bleu3": bleu3,
        "bleu4": bleu4,
        "meteor": meteor,
        "rougeL": rougeL
    }
