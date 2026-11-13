import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Plus
from src.data_loader import Chunk

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself",
    "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some",
    "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves"
}

def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens without stopwords."""
    tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class BM25BaselineRetriever:
    """Non-Transformer baseline retriever utilizing BM25Plus and extractive rule-based synthesis."""
    
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.corpus_tokens = [tokenize(f"{c.section_title} {c.content}") for c in chunks]
        self.bm25 = BM25Plus(self.corpus_tokens) if self.corpus_tokens else None

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Retrieve top_k chunks using BM25 scoring."""
        if not self.chunks or not self.bm25:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return [(self.chunks[i], 0.0) for i in range(min(top_k, len(self.chunks)))]
        
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked_indices]

    def generate_extractive_answer(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Synthesize an answer using heuristic rule-based extractive sentence matching."""
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return {
                "answer": "No relevant keyword matches found in documentation.",
                "retrieved_chunks": [],
                "scores": [],
                "method": "BM25_extractive_baseline"
            }

        query_tokens = set(tokenize(query))
        best_chunk, best_score = results[0]

        
        chunk_tokens = set(tokenize(f"{best_chunk.section_title} {best_chunk.content}"))
        if not query_tokens.intersection(chunk_tokens):
            return {
                "answer": "No relevant keyword matches found in documentation.",
                "retrieved_chunks": [r[0].chunk_id for r in results],
                "scores": [r[1] for r in results],
                "method": "BM25_extractive_baseline"
            }

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_chunk.content) if s.strip()]

        best_sent = ""
        highest_overlap = -1
        for sent in sentences:
            sent_set = set(tokenize(sent))
            overlap = len(query_tokens.intersection(sent_set))
            if overlap > highest_overlap and len(sent) > 15:
                highest_overlap = overlap
                best_sent = sent

        if not best_sent and sentences:
            best_sent = sentences[0]

        return {
            "answer": best_sent if best_sent else "No direct matching sentence extracted.",
            "retrieved_chunks": [r[0].chunk_id for r in results],
            "scores": [r[1] for r in results],
            "method": "BM25_extractive_baseline"
        }
