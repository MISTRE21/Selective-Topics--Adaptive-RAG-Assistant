import re
from typing import List, Dict, Any

class GroundingVerifier:
    """Verifies that the generated answer is faithful to and grounded in the retrieved documentation."""

    def __init__(self, min_grounding_ratio: float = 0.35):
        self.min_grounding_ratio = min_grounding_ratio

    def verify(self, answer: str, context_chunks: List[str]) -> Dict[str, Any]:
        if not context_chunks:
            return {
                "is_grounded": False,
                "grounding_score": 0.0,
                "reason": "No supporting context chunks provided."
            }

        
        answer_tokens = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", answer.lower()))
        if not answer_tokens:
            return {
                "is_grounded": True,
                "grounding_score": 1.0,
                "reason": "Empty or trivial response."
            }

        combined_context = " ".join(context_chunks).lower()
        context_tokens = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", combined_context))

        grounded_tokens = answer_tokens.intersection(context_tokens)
        grounding_score = len(grounded_tokens) / max(len(answer_tokens), 1)

        is_grounded = grounding_score >= self.min_grounding_ratio

        
        ungrounded = list(answer_tokens - context_tokens)[:8]

        return {
            "is_grounded": is_grounded,
            "grounding_score": round(grounding_score, 4),
            "matched_token_count": len(grounded_tokens),
            "total_token_count": len(answer_tokens),
            "unsupported_tokens": ungrounded,
            "reason": (
                "Response is strongly supported by documentation evidence."
                if is_grounded
                else f"Potential hallucination or weak grounding: only {len(grounded_tokens)} of {len(answer_tokens)} concepts verified in source documentation."
            )
        }
