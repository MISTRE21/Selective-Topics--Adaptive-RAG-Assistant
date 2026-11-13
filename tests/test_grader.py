from src.data_loader import Chunk
from src.adaptive_rag.grader import DocumentGrader

def test_grader_relevant():
    grader = DocumentGrader(score_threshold=0.3)
    chunk = Chunk("c1", "doc", "Security", "Use OAuth2PasswordBearer to get bearer tokens", 50)
    result = grader.grade_chunks("OAuth2 tokens", [(chunk, 0.8)])
    assert result["needs_rewrite"] is False
    assert len(result["relevant_chunks"]) == 1

def test_grader_irrelevant():
    grader = DocumentGrader(score_threshold=0.5)
    chunk = Chunk("c2", "doc", "Cooking", "Recipe for apple pie with cinnamon", 50)
    result = grader.grade_chunks("database session", [(chunk, 0.05)])
    assert result["needs_rewrite"] is True
    assert len(result["relevant_chunks"]) == 0
