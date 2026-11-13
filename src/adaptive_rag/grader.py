import re
from typing import List, Tuple, Dict, Any
from src.data_loader import Chunk

class DocumentGrader:
    """Evaluates relevance of retrieved chunks to the developer query,
    filtering out distracting noise and determining if query rewriting is required.
    """

    def __init__(self, score_threshold: float = 0.40):
        self.score_threshold = score_threshold

    def grade_chunks(self, query: str, scored_chunks: List[Tuple[Chunk, float]]) -> Dict[str, Any]:
        graded_results = []
        relevant_chunks: List[Tuple[Chunk, float]] = []

        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower()))

        for chunk, score in scored_chunks:
            content_tokens = set(re.findall(r"\b\w{3,}\b", chunk.content.lower()))
            overlap_ratio = len(query_tokens.intersection(content_tokens)) / max(len(query_tokens), 1)

            blended_grade = 0.7 * score + 0.3 * overlap_ratio
            is_relevant = blended_grade >= self.score_threshold

            graded_results.append({
                "chunk_id": chunk.chunk_id,
                "section": chunk.section_title,
                "raw_score": round(score, 4),
                "overlap_ratio": round(overlap_ratio, 4),
                "blended_grade": round(blended_grade, 4),
                "is_relevant": is_relevant
            })

            if is_relevant:
                relevant_chunks.append((chunk, blended_grade))

        needs_rewrite = len(relevant_chunks) == 0

        return {
            "graded_chunks": graded_results,
            "relevant_chunks": relevant_chunks,
            "needs_rewrite": needs_rewrite,
            "relevant_count": len(relevant_chunks)
        }
