import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
bench_file = PROJECT_ROOT / "reports" / "benchmark_results.json"
figs_dir = PROJECT_ROOT / "reports" / "figures"
figs_dir.mkdir(parents=True, exist_ok=True)

with open(bench_file, "r", encoding="utf-8") as f:
    data = json.load(f)

b = data["baseline_bm25"]
n = data["naive_rag"]
a = data["adaptive_rag"]

# Figure 1: Overall Retrieval & Evaluation Metrics
metrics = ["Hit@1", "Hit@3", "MRR@3", "Precision@3", "Recall@3"]
bm_vals = [b["mean_hit_at_1"], b["mean_hit_at_3"], b["mean_mrr_at_3"], b["mean_precision_at_3"], b["mean_recall_at_3"]]
nv_vals = [n["mean_hit_at_1"], n["mean_hit_at_3"], n["mean_mrr_at_3"], n["mean_precision_at_3"], n["mean_recall_at_3"]]
ad_vals = [a["mean_hit_at_1"], a["mean_hit_at_3"], a["mean_mrr_at_3"], a["mean_precision_at_3"], a["mean_recall_at_3"]]

x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(9, 5))
plt.bar(x - width, bm_vals, width, label="BM25 Baseline (Sparse)", color="#718096")
plt.bar(x, nv_vals, width, label="Naive Dense RAG", color="#4299E1")
plt.bar(x + width, ad_vals, width, label="Adaptive RAG (Ours)", color="#38A169")

plt.ylabel("Score", fontsize=12)
plt.title("Comparative Performance Across Held-Out Evaluation Metrics", fontsize=14, fontweight="bold")
plt.xticks(x, metrics, fontsize=11)
plt.ylim(0, 1.05)
plt.legend(frameon=True, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig1_path = figs_dir / "metric_comparison.png"
plt.savefig(fig1_path, dpi=300)
plt.close()
print(f"Saved: {fig1_path}")

# Figure 2: Category Breakdown (MRR@3)
cats = sorted(list(b["category_breakdown"].keys()))
clean_labels = [c.replace("_", " ").title() for c in cats]
bm_cat = [b["category_breakdown"][c]["mrr_at_3"] for c in cats]
nv_cat = [n["category_breakdown"][c]["mrr_at_3"] for c in cats]
ad_cat = [a["category_breakdown"][c]["mrr_at_3"] for c in cats]

x = np.arange(len(cats))
plt.figure(figsize=(11, 5.5))
plt.bar(x - width, bm_cat, width, label="BM25 Baseline", color="#718096")
plt.bar(x, nv_cat, width, label="Naive Dense RAG", color="#4299E1")
plt.bar(x + width, ad_cat, width, label="Adaptive RAG (Ours)", color="#38A169")

plt.ylabel("MRR@3 Score", fontsize=12)
plt.title("Retrieval Quality (MRR@3) by Developer Query Category", fontsize=14, fontweight="bold")
plt.xticks(x, clean_labels, fontsize=10, rotation=15)
plt.ylim(0, 1.15)
plt.legend(frameon=True, fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
fig2_path = figs_dir / "category_breakdown.png"
plt.savefig(fig2_path, dpi=300)
plt.close()
print(f"Saved: {fig2_path}")

print("Successfully generated all publication-ready figures!")
