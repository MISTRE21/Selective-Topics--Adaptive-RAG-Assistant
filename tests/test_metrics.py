from src.evaluation.metrics import EvaluationMetrics

def test_retrieval_metrics():
    metrics = EvaluationMetrics()
    retrieved = ["doc_a", "doc_b", "doc_c"]
    target = ["doc_b"]

    assert metrics.hit_at_k(retrieved, target, k=1) == 0.0
    assert metrics.hit_at_k(retrieved, target, k=3) == 1.0
    assert metrics.mrr_at_k(retrieved, target, k=3) == 0.5  
    assert metrics.precision_at_k(retrieved, target, k=3) == 1.0 / 3.0
    assert metrics.recall_at_k(retrieved, target, k=3) == 1.0

def test_rouge_and_bleu():
    metrics = EvaluationMetrics()
    ref = "FastAPI returns HTTP 422 error on validation failure."
    pred = "FastAPI returns an HTTP 422 Unprocessable Entity error on validation failure."

    rouge = metrics.rouge_scores(pred, ref)
    assert rouge["rougeL"] > 0.6
    bleu = metrics.bleu4_score(pred, ref)
    assert bleu > 0.3
