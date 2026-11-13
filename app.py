import sys
import json
import time
from pathlib import Path
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DocumentLoader
from src.baseline.bm25_retriever import BM25BaselineRetriever
from src.adaptive_rag.retriever import DenseRetriever
from src.adaptive_rag.pipeline import AdaptiveRAGPipeline

st.set_page_config(
    page_title="Adaptive RAG Assistant for Software Project Documentation",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_system():
    raw_docs_dir = PROJECT_ROOT / "data" / "raw_docs"
    loader = DocumentLoader(raw_docs_dir)
    chunks = loader.load_and_chunk()
    bm25 = BM25BaselineRetriever(chunks)
    dense = DenseRetriever(chunks)
    adaptive = AdaptiveRAGPipeline(chunks=chunks, retriever=dense)
    return chunks, bm25, dense, adaptive

chunks, bm25_retriever, dense_retriever, adaptive_pipeline = load_system()

st.title("🤖 Adaptive RAG Assistant for Software Project Documentation")
st.markdown(
    "A Transformer-powered Software Engineering assistant for navigating evolving API references, "
    "architecture patterns, and developer documentation with **Adaptive Query Routing**, "
    "**Document Relevance Grading**, **Self-Correction**, and **Hallucination Verification**."
)

tab1, tab2, tab3 = st.tabs(["💬 Interactive Q&A Assistant", "📊 Benchmark Comparison", "🔍 Documentation Corpus"])

with tab1:
    st.subheader("Developer Query Console")
    
    preset_queries = [
        "Select a preset test query...",
        "What parameter is used in FastAPI route decorators to specify a 201 response status?",
        "How can I mock or replace database connections in unit tests without changing endpoint code?",
        "How do dependencies with yield interact with database session closing and route-level error exceptions?",
        "How do I conceal internal model attributes from external API clients upon JSON serialization?",
        "How do you configure Hibernate entity relationships in FastAPI with @Entity and @Table?"
    ]

    selected_preset = st.selectbox("Quick-load test scenario:", preset_queries)
    default_text = "" if selected_preset == preset_queries[0] else selected_preset

    user_query = st.text_input("Enter your software engineering question:", value=default_text)

    col_btn1, col_btn2 = st.columns([1, 5])
    run_btn = col_btn1.button("🚀 Run Assistant", type="primary")

    if run_btn and user_query.strip():
        with st.spinner("Processing query through Adaptive RAG pipeline..."):
          
            t0 = time.time()
            bm25_result = bm25_retriever.generate_extractive_answer(user_query)
            bm25_lat = (time.time() - t0) * 1000

           
            adapt_result = adaptive_pipeline.run(user_query)

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### ⚡ Non-Transformer Baseline (BM25)")
            st.caption(f"Method: Inverted Term Frequency + Extractive Sentence Window | Latency: {bm25_lat:.2f} ms")
            st.info(bm25_result["answer"])
            with st.expander("BM25 Retrieval Details"):
                st.write("**Top Chunks Retrieved:**", bm25_result["retrieved_chunks"])
                st.write("**BM25 Scores:**", [round(s, 2) for s in bm25_result["scores"]])

        with col_right:
            st.markdown("### 🧠 Adaptive RAG Assistant (Transformer)")
            route_color = {
                "direct": "blue",
                "retrieve_standard": "green",
                "retrieve_multi_hop": "orange",
                "reject": "red"
            }.get(adapt_result["route"], "gray")
            
            st.caption(
                f"Route: :{route_color}[**{adapt_result['route'].upper()}**] | "
                f"Latency: {adapt_result['latency_ms']:.2f} ms | "
                f"Rewritten: {adapt_result['was_rewritten']}"
            )
            st.success(adapt_result["answer"])

           
            with st.expander("🛠️ Adaptive Pipeline Trace & Grounding"):
                st.markdown(f"**Routing Reason:** {adapt_result['route_reason']}")
                if adapt_result["was_rewritten"]:
                    st.warning(f"**Query Rewritten to:** {adapt_result['rewritten_query']}")

                st.markdown("**Retrieved Chunks & Relevance Grades:**")
                if adapt_result["graded_chunks"]:
                    st.table(adapt_result["graded_chunks"])
                else:
                    st.write("No retrieval performed (Direct or Negative Route).")

                st.markdown("**Hallucination & Grounding Check:**")
                g = adapt_result["grounding"]
                st.write(f"- Grounding Ratio: `{g.get('grounding_score', 0.0) * 100:.1f}%`")
                st.write(f"- Verified: `{g.get('is_grounded', False)}`")
                st.write(f"- Diagnostic: {g.get('reason', '')}")
                if g.get("unsupported_tokens"):
                    st.write(f"- Unsubstantiated tokens: `{g['unsupported_tokens']}`")

with tab2:
    st.subheader("Held-Out Evaluation Benchmark")
    st.markdown("Quantitative comparison on the unseen 25-query held-out evaluation suite:")
    bench_file = PROJECT_ROOT / "reports" / "benchmark_results.json"
    if bench_file.exists():
        with open(bench_file, "r", encoding="utf-8") as f:
            bdata = json.load(f)

        b = bdata["baseline_bm25"]
        n = bdata["naive_rag"]
        a = bdata["adaptive_rag"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Adaptive RAG MRR@3", f"{a['mean_mrr_at_3']:.3f}", f"{(a['mean_mrr_at_3'] - b['mean_mrr_at_3']):+.3f} vs BM25")
        col2.metric("Adaptive RAG Hit@3", f"{a['mean_hit_at_3'] * 100:.1f}%", f"{(a['mean_hit_at_3'] - b['mean_hit_at_3']) * 100:+.1f}% vs BM25")
        col3.metric("Adaptive RAG ROUGE-L", f"{a['mean_rougeL']:.3f}", f"{(a['mean_rougeL'] - b['mean_rougeL']):+.3f} vs BM25")

        summary_table = [
            {"Metric": "Hit@1", "BM25 Baseline": b["mean_hit_at_1"], "Naive RAG": n["mean_hit_at_1"], "Adaptive RAG": a["mean_hit_at_1"]},
            {"Metric": "Hit@3", "BM25 Baseline": b["mean_hit_at_3"], "Naive RAG": n["mean_hit_at_3"], "Adaptive RAG": a["mean_hit_at_3"]},
            {"Metric": "MRR@3", "BM25 Baseline": b["mean_mrr_at_3"], "Naive RAG": n["mean_mrr_at_3"], "Adaptive RAG": a["mean_mrr_at_3"]},
            {"Metric": "Precision@3", "BM25 Baseline": b["mean_precision_at_3"], "Naive RAG": n["mean_precision_at_3"], "Adaptive RAG": a["mean_precision_at_3"]},
            {"Metric": "Recall@3", "BM25 Baseline": b["mean_recall_at_3"], "Naive RAG": n["mean_recall_at_3"], "Adaptive RAG": a["mean_recall_at_3"]},
            {"Metric": "ROUGE-L F1", "BM25 Baseline": b["mean_rougeL"], "Naive RAG": n["mean_rougeL"], "Adaptive RAG": a["mean_rougeL"]},
            {"Metric": "BLEU-4", "BM25 Baseline": b["mean_bleu4"], "Naive RAG": n["mean_bleu4"], "Adaptive RAG": a["mean_bleu4"]},
            {"Metric": "Avg Latency (ms)", "BM25 Baseline": b["avg_latency_ms"], "Naive RAG": n["avg_latency_ms"], "Adaptive RAG": a["avg_latency_ms"]},
        ]
        st.dataframe(summary_table, use_container_width=True)

        st.markdown("### 📈 Evaluation Plots")
        fig1 = PROJECT_ROOT / "reports" / "figures" / "metric_comparison.png"
        fig2 = PROJECT_ROOT / "reports" / "figures" / "category_breakdown.png"
        p_col1, p_col2 = st.columns(2)
        if fig1.exists():
            p_col1.image(str(fig1), caption="Overall Metric Comparison", use_container_width=True)
        if fig2.exists():
            p_col2.image(str(fig2), caption="MRR@3 Breakdown by Category", use_container_width=True)
    else:
        st.warning("Benchmark results file not found. Run `python scripts/run_evaluation.py` to populate.")

with tab3:
    st.subheader("Indexed Documentation Corpus")
    st.markdown(f"Total semantic chunks loaded: **{len(chunks)}**")
    for c in chunks:
        with st.expander(f"📄 {c.chunk_id} ({c.char_length} chars)"):
            st.markdown(f"**Section:** {c.section_title}")
            st.code(c.content, language="markdown")
