from pathlib import Path
from src.data_loader import DocumentLoader, Chunk, slugify

def test_slugify():
    assert slugify("Path Parameters") == "path-parameters"
    assert slugify("OAuth2 & JWT Auth!") == "oauth2-jwt-auth"
    assert slugify("  Multiple   Spaces  ") == "multiple-spaces"

def test_chunk_markdown():
    sample_md = """# Sample Title
## First Section
This is some content.
```python
def foo():
    pass
```
## Second Section
More details here.
"""
    loader = DocumentLoader(Path("."))
    chunks = loader.chunk_markdown("sample.md", sample_md)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == "sample.md#first-section"
    assert "def foo():" in chunks[0].content
    assert chunks[1].chunk_id == "sample.md#second-section"
