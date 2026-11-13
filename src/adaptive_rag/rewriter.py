import re

SYNONYM_MAPPINGS = {
    "conceal": "filter hide response_model",
    "private attributes": "sensitive data fields UserOut",
    "tear down": "cleanup yield finally close",
    "teardown": "cleanup yield finally close",
    "execution duration": "process time header middleware",
    "after sending response": "background tasks BackgroundTasks",
    "mock database": "dependency_overrides TestClient",
    "replace database": "dependency_overrides TestClient",
    "restrict browser": "CORS CORSMiddleware allow_origins",
    "outbound": "response_model output response"
}

BOILERPLATE_PATTERNS = [
    r"^(how do i|how can i|how to|what is the way to|tell me how to)\s+",
    r"^(can you explain|please explain|show me how to)\s+"
]

class QueryRewriter:
    """Reformulates queries when initial retrieval results fail relevance grading."""

    def rewrite(self, query: str) -> str:
        rewritten = query.lower().strip()

        
        for pattern in BOILERPLATE_PATTERNS:
            rewritten = re.sub(pattern, "", rewritten, flags=re.IGNORECASE).strip()

        
        added_terms = []
        for term, expansion in SYNONYM_MAPPINGS.items():
            if term in rewritten:
                added_terms.append(expansion)

        if added_terms:
            rewritten = f"{rewritten} {' '.join(added_terms)}"

        return rewritten.strip()
