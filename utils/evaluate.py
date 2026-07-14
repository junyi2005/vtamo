from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU, CHRF, TER


def evaluate_results(predictions, references, split="train", device='cpu', tokenizer='13a'):
    """
    Evaluate prediction results using BLEU and ROUGE metrics.

    Args:
        predictions (list): List of predicted sequences.
        references (list): List of reference sequences.
        tokenizer (object, optional): Tokenizer if needed for evaluation.
        split (str): The data split being evaluated.

    Returns:
        dict: A dictionary of evaluation scores.
    """
    log_dicts = {}

    # Always calculate BLEU1-4 for both val and test
    for i in range(1, 5):
        score = BLEU(max_ngram_order=i, tokenize=tokenizer).corpus_score(predictions, [references]).score
        log_dicts[f"{split}/bleu{i}"] = score

    # Always calculate ROUGE-L score for both val and test
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'] for ref, pred in zip(references, predictions)]

    # Aggregate ROUGE-L scores (average precision, recall, and F1)
    avg_precision = sum(score.precision for score in rouge_scores) / len(rouge_scores)
    avg_recall = sum(score.recall for score in rouge_scores) / len(rouge_scores)
    avg_f1 = sum(score.fmeasure for score in rouge_scores) / len(rouge_scores)

    log_dicts[f"{split}/rougeL_precision"] = avg_precision
    log_dicts[f"{split}/rougeL_recall"] = avg_recall
    log_dicts[f"{split}/rougeL_f1"] = avg_f1

    return log_dicts