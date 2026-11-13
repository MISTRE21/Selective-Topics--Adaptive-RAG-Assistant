import re
from typing import Dict, Any, List

ROUTE_DIRECT = "direct"
ROUTE_RETRIEVE_STANDARD = "retrieve_standard"
ROUTE_RETRIEVE_MULTI_HOP = "retrieve_multi_hop"
ROUTE_REJECT = "reject"

OUT_OF_DOMAIN_KEYWORDS = [
    "django", "hibernate", "spring boot", "react jsx server-side",
    "kubernetes pod daemon", "@app.k8s_sidecar", "deprecated_cache_resolver",
    "angular", "vue", "laravel", "rails", "asp.net"
]

MULTI_HOP_INDICATORS = [
    "both", "and also", "interact with", "simultaneously", "when should",
    "instead of", "difference from", "combined with", "across", "interplay"
]

class QueryRouter:
    """Adaptive query router that classifies incoming developer queries into:
    - direct: queries answerable directly without document retrieval
    - retrieve_standard: single-topic API or documentation lookup
    - retrieve_multi_hop: cross-component or comparative queries requiring iterative retrieval
    - reject: out-of-domain or unanswerable queries
    """

    def __init__(self, out_of_domain_keywords: List[str] = None):
        self.out_of_domain_keywords = out_of_domain_keywords or OUT_OF_DOMAIN_KEYWORDS

    def route(self, query: str) -> Dict[str, Any]:
        q_lower = query.lower().strip()

       
        for bad_kw in self.out_of_domain_keywords:
            if bad_kw in q_lower:
                return {
                    "route": ROUTE_REJECT,
                    "confidence": 0.95,
                    "reason": f"Query contains out-of-domain or non-existent framework concept: '{bad_kw}'"
                }

        
        for ind in MULTI_HOP_INDICATORS:
            if ind in q_lower:
                return {
                    "route": ROUTE_RETRIEVE_MULTI_HOP,
                    "confidence": 0.88,
                    "reason": f"Query involves multiple concepts or comparison indicated by '{ind}'"
                }

        
        if q_lower in ["hi", "hello", "what is python", "help"]:
            return {
                "route": ROUTE_DIRECT,
                "confidence": 0.99,
                "reason": "Direct general query"
            }

        
        return {
            "route": ROUTE_RETRIEVE_STANDARD,
            "confidence": 0.90,
            "reason": "Standard single-domain documentation retrieval"
        }
