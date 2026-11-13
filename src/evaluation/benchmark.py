import json
import time
from pathlib import Path
from typing import List, Dict, Any

from src.data_loader import Chunk
from src.baseline.bm25_retriever import BM25BaselineRetriever
from src.adaptive_rag.retriever import DenseRetriever
from src.adaptive_rag.pipeline import AdaptiveRAGPipeline
from src.evaluation.metrics import EvaluationMetrics
from src.models.generator import LLMGenerator

class BenchmarkRunner:
    """Orchestrates comprehensive comparative evaluation across Baseline, Naive RAG, and Adaptive RAG."""

    def __init__(self, chunks: List[Chunk], test_data_path: Path):
        self.chunks = chunks
        self.test_data = json.loads(Path(test_data_path).read_text(encoding="utf-8"))
        self.metrics = EvaluationMetrics()

        
        self.bm25_baseline = BM25BaselineRetriever(chunks)
        self.dense_retriever = DenseRetriever(chunks)
        self.generator = LLMGenerator()
        self.adaptive_pipeline = AdaptiveRAGPipeline(
            chunks=chunks,
            retriever=self.dense_retriever,
            generator=self.generator
        )

    def run_all(self) -> Dict[str, Any]:
        baseline_results = []
        naive_rag_results = []
        adaptive_rag_results = []

        print(f"Beginning benchmark run across {len(self.test_data)} test items...")

        for idx, item in enumerate(self.test_data):
            qid = item["id"]
            query = item["query"]
            targets = item["relevant_doc_ids"]
            reference = item["ground_truth_answer"]
            category = item["category"]

            
            t0 = time.time()
            bm25_res = self.bm25_baseline.retrieve(query, top_k=3)
            bm25_ans_dict = self.bm25_baseline.generate_extractive_answer(query, top_k=3)
            bm25_lat = (time.time() - t0) * 1000
            bm25_retrieved_ids = [c[0].chunk_id for c in bm25_res]
            bm25_answer = bm25_ans_dict["answer"]

            baseline_results.append(self._evaluate_single(
                qid, category, bm25_retrieved_ids, targets, bm25_answer, reference, bm25_lat
            ))

            
            t0 = time.time()
            dense_res = self.dense_retriever.retrieve(query, top_k=3)
            dense_retrieved_ids = [c[0].chunk_id for c in dense_res]
            context_texts = [f"{c[0].section_title}\n{c[0].content}" for c in dense_res]
            naive_answer = self.generator.generate(prompt=query, context_chunks=context_texts, query=query)
            naive_lat = (time.time() - t0) * 1000

            naive_rag_results.append(self._evaluate_single(
                qid, category, dense_retrieved_ids, targets, naive_answer, reference, naive_lat
            ))

            
            adapt_res = self.adaptive_pipeline.run(query)
            adapt_retrieved_ids = adapt_res["retrieved_chunk_ids"]
            adapt_answer = adapt_res["answer"]
            adapt_lat = adapt_res["latency_ms"]

            eval_entry = self._evaluate_single(
                qid, category, adapt_retrieved_ids, targets, adapt_answer, reference, adapt_lat
            )
            eval_entry["route"] = adapt_res["route"]
            eval_entry["was_rewritten"] = adapt_res["was_rewritten"]
            eval_entry["grounding_score"] = adapt_res["grounding"].get("grounding_score", 0.0)
            eval_entry["is_grounded"] = adapt_res["grounding"].get("is_grounded", False)
            adaptive_rag_results.append(eval_entry)

        summary = {
            "num_test_queries": len(self.test_data),
            "baseline_bm25": self._aggregate(baseline_results),
            "naive_rag": self._aggregate(naive_rag_results),
            "adaptive_rag": self._aggregate(adaptive_rag_results),
            "detailed_records": {
                "baseline": baseline_results,
                "naive_rag": naive_rag_results,
                "adaptive_rag": adaptive_rag_results
            }
        }
        return summary

    def _evaluate_single(
        self, qid: str, category: str, retrieved_ids: List[str], targets: List[str],
        prediction: str, reference: str, latency_ms: float
    ) -> Dict[str, Any]:
        rouge = self.metrics.rouge_scores(prediction, reference)
        bleu = self.metrics.bleu4_score(prediction, reference)
        hit1 = self.metrics.hit_at_k(retrieved_ids, targets, k=1)
        hit3 = self.metrics.hit_at_k(retrieved_ids, targets, k=3)
        mrr3 = self.metrics.mrr_at_k(retrieved_ids, targets, k=3)
        p3 = self.metrics.precision_at_k(retrieved_ids, targets, k=3)
        r3 = self.metrics.recall_at_k(retrieved_ids, targets, k=3)

        return {
            "id": qid,
            "category": category,
            "hit_at_1": hit1,
            "hit_at_3": hit3,
            "mrr_at_3": round(mrr3, 4),
            "precision_at_3": round(p3, 4),
            "recall_at_3": round(r3, 4),
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"],
            "bleu4": bleu,
            "latency_ms": round(latency_ms, 2),
            "prediction": prediction,
            "retrieved_chunk_ids": retrieved_ids
        }

    def _aggregate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(records)
        if n == 0:
            return {}

        avg = {
            "mean_hit_at_1": round(sum(r["hit_at_1"] for r in records) / n, 4),
            "mean_hit_at_3": round(sum(r["hit_at_3"] for r in records) / n, 4),
            "mean_mrr_at_3": round(sum(r["mrr_at_3"] for r in records) / n, 4),
            "mean_precision_at_3": round(sum(r["precision_at_3"] for r in records) / n, 4),
            "mean_recall_at_3": round(sum(r["recall_at_3"] for r in records) / n, 4),
            "mean_rouge1": round(sum(r["rouge1"] for r in records) / n, 4),
            "mean_rougeL": round(sum(r["rougeL"] for r in records) / n, 4),
            "mean_bleu4": round(sum(r["bleu4"] for r in records) / n, 4),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 2)
        }

        
        categories = sorted(list(set(r["category"] for r in records)))
        cat_breakdown = {}
        for cat in categories:
            cat_recs = [r for r in records if r["category"] == cat]
            cn = len(cat_recs)
            cat_breakdown[cat] = {
                "count": cn,
                "hit_at_3": round(sum(r["hit_at_3"] for r in cat_recs) / cn, 4),
                "mrr_at_3": round(sum(r["mrr_at_3"] for r in cat_recs) / cn, 4),
                "rougeL": round(sum(r["rougeL"] for r in cat_recs) / cn, 4)
            }
        avg["category_breakdown"] = cat_breakdown

        return avg
