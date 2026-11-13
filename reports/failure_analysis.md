# Failure Analysis: Adaptive RAG Assistant for Software Documentation

## Executive Summary
This empirical failure analysis examines qualitative and quantitative error modes observed across the held-out test benchmark, comparing the **Non-Transformer BM25 Baseline**, **Naive Dense RAG**, and the **Adaptive RAG Assistant**.

## Taxonomy of Failure Modes

| Failure Category | Primary Root Cause | Affected Pipelines | Frequency | Severity |
|---|---|---|---|---|
| **Type 1: Lexical Mismatch / Vocabulary Gap** | Developer uses semantic synonyms not literally present in doc tokens | BM25 Baseline | High | High |
| **Type 2: Confabulation on Negative Queries** | Model attempts to answer queries for non-existent / out-of-domain features | Naive Dense RAG | High | Critical |
| **Type 3: Multi-Hop Decomposition Failure** | Query spans disparate subsystems; single-step retrieval misses context | Naive Dense RAG, BM25 | Moderate | Medium |
| **Type 4: Extractive Window Truncation** | Rule-based baseline clips code or sentences mid-thought | BM25 Baseline | High | Medium |
| **Type 5: Semantic Drift / Low Specificity** | Retrieval yields generic parent context rather than specific method signature | All Pipelines | Low | Low |

---

## Deep-Dive Case Studies

### Case 1: Lexical Gap - How do I conceal internal model attributes from external API clients upon JSON serialization?
- **Query ID**: `test_16` (adversarial_paraphrase)
- **Developer Query**: *"How do I conceal internal model attributes from external API clients upon JSON serialization?"*
- **Ground Truth**: Use `response_model` on the path decorator (such as `response_model=UserOut`) to automatically filter out private attributes from output responses.
- **BM25 Baseline Output**: Dependency Injection (DI) allows endpoints to declare dependencies required for business logic, database sessions, authentication, or external API clients. *(Retrieved: ['03_dependency_injection.md#what-is-dependency-injection', '01_routing_and_parameters.md#response-model-and-filtering', '01_routing_and_parameters.md#overview'])*
- **Adaptive RAG Output**: FastAPI: Routing and Parameters - Response Model and Filtering
### Context: FastAPI: Routing and Parameters > Response Model and Filtering

Use `response_model` to filter sensitive data from output responses:

```python
@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user  # UserOut filters out password_hash automatically
```



```python
@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user  # UserOut filters out password_hash automatically
``` *(Route: retrieve_standard, Rewritten: True)*
- **Root Cause Analysis**: The query utilizes vocabulary like *'conceal internal model attributes'* or *'restrict browser requests'*. The BM25 baseline relies purely on inverted term frequencies; since the documentation uses terms like *'response_model'* and *'CORSMiddleware'*, BM25 experiences total recall failure (Hit@3 = 0). Conversely, Adaptive RAG leverages dense transformer representations and synonym rewriting to project these developer queries directly to the correct semantic cluster.
- **Mitigation**: Dense sentence-transformers embeddings with adaptive query reformulation.

### Case 2: Out-of-Domain Hallucination - How do you configure Hibernate entity relationships in FastAPI with @Entity and @Table?
- **Query ID**: `test_21` (unanswerable_negative)
- **Developer Query**: *"How do you configure Hibernate entity relationships in FastAPI with @Entity and @Table?"*
- **Ground Truth**: This feature does not exist in FastAPI documentation. Hibernate and @Entity are Java frameworks, not FastAPI components.
- **Naive RAG Behavior**: Naive RAG unconditionally retrieves nearest neighbor chunks (despite low absolute relevance) and forces the generator to construct an answer, causing confabulation of non-existent decorators or parameters.
- **Adaptive RAG Output**: I cannot find this feature in the official documentation. Query contains out-of-domain or non-existent framework concept: 'hibernate'. FastAPI does not provide or support this functionality natively. *(Route: reject, Grounded: True)*
- **Root Cause Analysis**: Standard RAG pipelines lack an admission control filter. When faced with an adversarial or unsupported framework feature (e.g. `@app.k8s_sidecar` or Hibernate), nearest-neighbor search still yields chunks with arbitrary cosine similarity. Adaptive RAG solves this at two levels: (1) Front-end Router classifies out-of-domain queries into `reject`, and (2) Document Relevance Grader flags candidate chunks below relevance thresholds.
- **Mitigation**: Query classification routing and two-tier relevance grading before context ingestion.

### Case 3: Multi-Hop Cross-Module Query - How do dependencies with yield interact with database session closing and route-level error exceptions?
- **Query ID**: `test_11` (multi_hop)
- **Developer Query**: *"How do dependencies with yield interact with database session closing and route-level error exceptions?"*
- **Ground Truth**: In `yield` dependencies, the teardown code in the `finally` block runs after the response or after an exception is caught, ensuring the database session closes even if an HTTPException is raised.
- **BM25 Retrieval**: `['03_dependency_injection.md#dependencies-with-yield-context-managers-db-sessions', '03_dependency_injection.md#global-and-router-level-dependencies', '08_testing_and_overrides.md#overriding-dependencies-during-testing']` (Recall@3: 0.5)
- **Adaptive RAG Output**: This allows code before `yield` to run as setup, and code after `yield` to run as teardown even if exceptions occur during request handling:

```python
from fastapi import Depends

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
``` FastAPI: Dependency Injection - Dependencies with `yield` (Context Managers / DB Sessions)
### Context: FastAPI: Dependency Injection > Dependencies with `yield` (Context Managers / DB Sessions)

FastAPI supports dependencies that use `yield` instead of `return`.



```python
from fastapi import Depends

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/{user_id}")
def read_user(user_id: int, db: Session = Depends(get_db)):
    return db.query(User).filter(User.id == user_id).first()
``` *(Route: retrieve_multi_hop, Retrieved: `['03_dependency_injection.md#dependencies-with-yield-context-managers-db-sessions']`)*
- **Root Cause Analysis**: The query spans two distinct concepts: e.g. dependency injection lifecycle vs. error handling / background tasks. A single-step dense or sparse retriever retrieves chunks for only the dominant keyword cluster, dropping the secondary prerequisite. The Adaptive RAG router detects `retrieve_multi_hop` and decomposes the query into sub-aspects, assembling evidence across both documentation sections.
- **Mitigation**: Query decomposition and multi-hop candidate merging.

## Summary of Engineering Recommendations
1. **Never Deploy Naive Unconditional RAG**: Always guard documentation assistants with an upfront intent router and document relevance grader to prevent hallucinations on negative or out-of-domain developer queries.
2. **Hybrid Semantic + Lexical Reranking**: While dense embeddings eliminate the vocabulary gap, exact keyword preservation remains crucial for exact API function names (e.g., `OAuth2PasswordBearer`). Combining dense vector similarity with token-level grading provides the highest precision.
3. **Decompose Multi-Concept Architectural Queries**: Complex software questions often span multiple architectural layers. Decomposing compound queries into sub-queries avoids blind spots in documentation retrieval.