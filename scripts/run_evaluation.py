import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DocumentLoader
from src.evaluation.benchmark import BenchmarkRunner

def main():
    print("=" * 70)
    print("  ADAPTIVE RAG ASSISTANT: HELD-OUT EVALUATION BENCHMARK")
    print("=" * 70)

    raw_docs_dir = PROJECT_ROOT / "data" / "raw_docs"
    loader = DocumentLoader(raw_docs_dir)
    chunks = loader.load_and_chunk()
    print(f"Loaded and chunked {len(chunks)} document chunks from {raw_docs_dir}")

    # Save chunks to processed
    processed_file = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    loader.save_processed_chunks(chunks, processed_file)
    print(f"Saved processed chunks to {processed_file}")

    test_file = PROJECT_ROOT / "data" / "benchmarks" / "held_out_test.json"
    runner = BenchmarkRunner(chunks, test_file)

    results = runner.run_all()

    # Save output to reports
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Comparative Summary Table
    print("\n" + "=" * 70)
    print(f"{'METRIC':<25} | {'BM25 BASELINE':<14} | {'NAIVE RAG':<12} | {'ADAPTIVE RAG':<12}")
    print("-" * 70)
    b = results["baseline_bm25"]
    n = results["naive_rag"]
    a = results["adaptive_rag"]

    metrics_to_print = [
        ("Hit@1", "mean_hit_at_1"),
        ("Hit@3", "mean_hit_at_3"),
        ("MRR@3", "mean_mrr_at_3"),
        ("Precision@3", "mean_precision_at_3"),
        ("Recall@3", "mean_recall_at_3"),
        ("ROUGE-1 F1", "mean_rouge1"),
        ("ROUGE-L F1", "mean_rougeL"),
        ("BLEU-4", "mean_bleu4"),
        ("Avg Latency (ms)", "avg_latency_ms"),
    ]

    for label, key in metrics_to_print:
        print(f"{label:<25} | {b[key]:<14} | {n[key]:<12} | {a[key]:<12}")
    print("=" * 70)

    print("\nCATEGORY BREAKDOWN (MRR@3 Comparison):")
    print("-" * 70)
    print(f"{'CATEGORY':<25} | {'BM25 BASELINE':<14} | {'NAIVE RAG':<12} | {'ADAPTIVE RAG':<12}")
    print("-" * 70)
    for cat in b["category_breakdown"]:
        bm_mrr = b["category_breakdown"][cat]["mrr_at_3"]
        nv_mrr = n["category_breakdown"][cat]["mrr_at_3"]
        ad_mrr = a["category_breakdown"][cat]["mrr_at_3"]
        print(f"{cat:<25} | {bm_mrr:<14} | {nv_mrr:<12} | {ad_mrr:<12}")
    print("=" * 70)
    print(f"\nDetailed benchmark results saved to: {out_file}\n")

if __name__ == "__main__":
    main()
