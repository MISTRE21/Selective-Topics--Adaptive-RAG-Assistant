from src.data_loader import Chunk
from src.adaptive_rag.pipeline import AdaptiveRAGPipeline

def test_pipeline_rejection():
    chunks = [
        Chunk("c1", "doc", "Routing", "Route handlers in FastAPI", 50)
    ]
    pipeline = AdaptiveRAGPipeline(chunks=chunks)
    res = pipeline.run("How do I configure Hibernate entities in FastAPI?")
    assert res["route"] == "reject"
    assert "FastAPI does not provide" in res["answer"]
    assert res["grounding"]["is_grounded"] is True

def test_pipeline_standard_retrieval():
    chunks = [
        Chunk("c1#status", "doc", "Status Codes", "Use status_code=status.HTTP_201_CREATED on POST endpoints.", 80)
    ]
    pipeline = AdaptiveRAGPipeline(chunks=chunks)
    res = pipeline.run("What parameter sets a 201 response status in FastAPI?")
    assert res["route"] == "retrieve_standard"
    assert len(res["retrieved_chunk_ids"]) > 0
    assert "status_code" in res["answer"]
    assert res["latency_ms"] >= 0.0
