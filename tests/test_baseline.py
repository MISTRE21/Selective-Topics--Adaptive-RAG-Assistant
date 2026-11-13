from src.data_loader import Chunk
from src.baseline.bm25_retriever import BM25BaselineRetriever, tokenize

def test_tokenize():
    tokens = tokenize("How to use FastAPI with OAuth2 tokens?")
    assert "fastapi" in tokens
    assert "oauth2" in tokens
    assert "how" not in tokens  

def test_bm25_retrieval():
    chunks = [
        Chunk("doc1#a", "doc1", "Auth", "OAuth2PasswordBearer authenticates requests using bearer tokens.", 100),
        Chunk("doc2#b", "doc2", "Database", "Connect to PostgreSQL using SessionLocal and yield dependencies.", 100)
    ]
    bm25 = BM25BaselineRetriever(chunks)
    res = bm25.retrieve("OAuth2 bearer tokens", top_k=1)
    assert len(res) == 1
    assert res[0][0].chunk_id == "doc1#a"
    assert res[0][1] > 0.0

def test_bm25_extractive_answer():
    chunks = [
        Chunk("doc1#a", "doc1", "Status", "Use status_code=201 to specify created status in route decorators.", 100)
    ]
    bm25 = BM25BaselineRetriever(chunks)
    ans = bm25.generate_extractive_answer("status_code 201")
    assert "201" in ans["answer"]
    assert ans["method"] == "BM25_extractive_baseline"
