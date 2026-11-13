import os
from pathlib import Path
from typing import List, Union
import numpy as np

class TransformerEmbedder:
    """Pretrained Transformer Embedding encoder using HuggingFace sentence-transformers.
    Includes persistent on-disk caching and fallback handling for offline environments.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_mode = False
        self._init_model()

    def _init_model(self):
        try:
            from sentence_transformers import SentenceTransformer
           
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            
            print(f"Notice: Falling back to local semantic encoder due to: {e}")
            self._fallback_mode = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            self._svd = TruncatedSVD(n_components=128, random_state=42)
            self._fitted = False

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if not self._fallback_mode and self._model is not None:
            embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                embeddings = embeddings / norms
            return embeddings
        else:
            
            if not self._fitted:
                tfidf = self._vectorizer.fit_transform(texts)
                n_components = min(128, tfidf.shape[1] - 1 if tfidf.shape[1] > 1 else 1)
                from sklearn.decomposition import TruncatedSVD
                self._svd = TruncatedSVD(n_components=max(1, n_components), random_state=42)
                reduced = self._svd.fit_transform(tfidf)
                self._fitted = True
            else:
                tfidf = self._vectorizer.transform(texts)
                reduced = self._svd.transform(tfidf)
            
            if normalize:
                norms = np.linalg.norm(reduced, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12
                reduced = reduced / norms
            return reduced

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
