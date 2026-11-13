import time
from typing import List, Dict, Any, Optional

from src.data_loader import Chunk
from src.adaptive_rag.router import (
    QueryRouter,
    ROUTE_DIRECT,
    ROUTE_RETRIEVE_STANDARD,
    ROUTE_RETRIEVE_MULTI_HOP,
    ROUTE_REJECT
)
from src.adaptive_rag.retriever import DenseRetriever
from src.adaptive_rag.grader import DocumentGrader
from src.adaptive_rag.rewriter import QueryRewriter
from src.adaptive_rag.verifier import GroundingVerifier
from src.models.generator import LLMGenerator

class AdaptiveRAGPipeline:
    """End-to-end Adaptive RAG Assistant pipeline.
    Orchestrates routing, dense retrieval, relevance grading, query rewriting,
    grounded generation, and hallucination verification.
    """

    def __init__(
        self,
        chunks: List[Chunk],
        retriever: Optional[DenseRetriever] = None,
        router: Optional[QueryRouter] = None,
        grader: Optional[DocumentGrader] = None,
        rewriter: Optional[QueryRewriter] = None,
        generator: Optional[LLMGenerator] = None,
        verifier: Optional[GroundingVerifier] = None,
        top_k: int = 3
    ):
        self.chunks = chunks
        self.retriever = retriever or DenseRetriever(chunks)
        self.router = router or QueryRouter()
        self.grader = grader or DocumentGrader()
        self.rewriter = rewriter or QueryRewriter()
        self.generator = generator or LLMGenerator()
        self.verifier = verifier or GroundingVerifier()
        self.top_k = top_k

    def run(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        trace: Dict[str, Any] = {
            "query": query,
            "latency_ms": 0.0,
            "route": None,
            "route_reason": None,
            "was_rewritten": False,
            "rewritten_query": None,
            "retrieved_chunk_ids": [],
            "retrieval_scores": [],
            "graded_chunks": [],
            "grounding": {},
            "answer": ""
        }

        route_decision = self.router.route(query)
        route = route_decision["route"]
        trace["route"] = route
        trace["route_reason"] = route_decision["reason"]

        
        if route == ROUTE_REJECT:
            trace["answer"] = (
                f"I cannot find this feature in the official documentation. {route_decision['reason']}. "
                "FastAPI does not provide or support this functionality natively."
            )
            trace["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            trace["grounding"] = {"is_grounded": True, "grounding_score": 1.0, "reason": "Accurate negative rejection"}
            return trace

        
        if route == ROUTE_DIRECT:
            trace["answer"] = "Hello! I am your Adaptive RAG assistant for Software Project Documentation. How can I help you navigate the codebase or API?"
            trace["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            trace["grounding"] = {"is_grounded": True, "grounding_score": 1.0, "reason": "Direct conversational response"}
            return trace

        
        if route == ROUTE_RETRIEVE_MULTI_HOP:
            candidates = self.retriever.retrieve_multi_hop(query, top_k=self.top_k + 1)
        else:
            candidates = self.retriever.retrieve(query, top_k=self.top_k)

        
        grading_result = self.grader.grade_chunks(query, candidates)
        trace["graded_chunks"] = grading_result["graded_chunks"]

        
        if grading_result["needs_rewrite"]:
            rewritten_q = self.rewriter.rewrite(query)
            trace["was_rewritten"] = True
            trace["rewritten_query"] = rewritten_q
            candidates = self.retriever.retrieve(rewritten_q, top_k=self.top_k)
            grading_result = self.grader.grade_chunks(rewritten_q, candidates)

        selected_chunks = [c for c, _ in grading_result["relevant_chunks"]]
        if not selected_chunks and candidates:
            
            selected_chunks = [candidates[0][0]]

        trace["retrieved_chunk_ids"] = [c.chunk_id for c in selected_chunks]
        trace["retrieval_scores"] = [round(s, 4) for c, s in candidates if c in selected_chunks]

        
        context_texts = [f"{c.section_title}\n{c.content}" for c in selected_chunks]
        raw_answer = self.generator.generate(
            prompt=query,
            context_chunks=context_texts,
            query=query
        )

        
        grounding_report = self.verifier.verify(raw_answer, context_texts)
        trace["grounding"] = grounding_report
        trace["answer"] = raw_answer
        trace["latency_ms"] = round((time.time() - start_time) * 1000, 2)

        return trace
