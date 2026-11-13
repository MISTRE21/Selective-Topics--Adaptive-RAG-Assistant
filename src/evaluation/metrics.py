import re
from typing import List, Dict, Any
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class EvaluationMetrics:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        self.smoother = SmoothingFunction().method1

    @staticmethod
    def hit_at_k(retrieved_ids: List[str], target_ids: List[str], k: int = 3) -> float:
        """Calculates Hit@k (1.0 if any target doc is in top k retrieved, else 0.0)."""
        if not target_ids:
            
            return 1.0 if len(retrieved_ids) == 0 else 0.0
        
        top_k_retrieved = set(retrieved_ids[:k])
        for target in target_ids:
            if target in top_k_retrieved:
                return 1.0
        return 0.0

    @staticmethod
    def mrr_at_k(retrieved_ids: List[str], target_ids: List[str], k: int = 3) -> float:
        """Calculates Mean Reciprocal Rank (MRR@k)."""
        if not target_ids:
            return 1.0 if len(retrieved_ids) == 0 else 0.0

        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in target_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], target_ids: List[str], k: int = 3) -> float:
        """Calculates Precision@k."""
        if not target_ids:
            return 1.0 if len(retrieved_ids) == 0 else 0.0

        sub = retrieved_ids[:k]
        if not sub:
            return 0.0
        matches = sum(1 for doc_id in sub if doc_id in target_ids)
        return matches / len(sub)

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], target_ids: List[str], k: int = 3) -> float:
        """Calculates Recall@k."""
        if not target_ids:
            return 1.0 if len(retrieved_ids) == 0 else 0.0

        sub = set(retrieved_ids[:k])
        matches = sum(1 for doc_id in target_ids if doc_id in sub)
        return matches / len(target_ids)

    def rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculates ROUGE-1 and ROUGE-L F1 scores."""
        scores = self.rouge.score(reference, prediction)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4)
        }

    def bleu4_score(self, prediction: str, reference: str) -> float:
        """Calculates sentence BLEU-4 score with smoothing."""
        ref_tokens = re.findall(r"\b\w+\b", reference.lower())
        pred_tokens = re.findall(r"\b\w+\b", prediction.lower())

        if not pred_tokens or not ref_tokens:
            return 0.0

        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=self.smoother)
        return round(float(score), 4)
