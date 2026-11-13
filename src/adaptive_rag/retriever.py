import os
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from src.data_loader import Chunk
from src.models.embeddings import TransformerEmbedder

class DenseRetriever:
    """Dense semantic retriever using pretrained Transformer embeddings."""

    def __init__(self, chunks: List[Chunk], embedder: Optional[TransformerEmbedder] = None, cache_dir: Optional[Path] = None):
        self.chunks = chunks
        self.embedder = embedder or TransformerEmbedder()
        self.cache_dir = cache_dir
        self.doc_embeddings: np.ndarray = np.array([])
        self._build_index()

    def _build_index(self):
        
        texts = [f"{c.section_title}\n{c.content}" for c in self.chunks]
        self.doc_embeddings = self.embedder.encode(texts, normalize=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Perform dense semantic similarity search."""
        if len(self.chunks) == 0:
            return []

        query_vec = self.embedder.encode_query(query)
        
        sim_scores = np.dot(self.doc_embeddings, query_vec)
        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        results = [(self.chunks[i], float(sim_scores[i])) for i in top_indices]
        return results

    def retrieve_multi_hop(self, query: str, top_k: int = 4) -> List[Tuple[Chunk, float]]:
        """Decompose complex queries into sub-aspects, retrieve candidates, and merge."""
        sub_queries = [query]
        
        delimiters = [r"\s+and\s+", r"\s+with\s+", r"\s+instead of\s+", r"\s+interact with\s+"]
        for delim in delimiters:
            parts = re.split(delim, query, flags=re.IGNORECASE)
            if len(parts) > 1:
                sub_queries.extend([p.strip() for p in parts if len(p.strip()) > 10])

        merged_results: Dict[str, Tuple[Chunk, float]] = {}
        for sq in sub_queries:
            sub_res = self.retrieve(sq, top_k=top_k)
            for chunk, score in sub_res:
                if chunk.chunk_id not in merged_results or score > merged_results[chunk.chunk_id][1]:
                    merged_results[chunk.chunk_id] = (chunk, score)

        
        sorted_candidates = sorted(merged_results.values(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_candidates
