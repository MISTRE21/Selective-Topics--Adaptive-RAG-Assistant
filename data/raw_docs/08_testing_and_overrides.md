# FastAPI: Testing and Dependency Overrides

## TestClient
FastAPI provides `TestClient`, based on `httpx`, which allows testing endpoints without running an HTTP network server:

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "Hello World"}

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}
```

## Overriding Dependencies during Testing
One of FastAPI's most powerful features is `app.dependency_overrides`. You can swap production database sessions or external auth services with mocks:

```python
from fastapi.testclient import TestClient

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

def test_get_users():
    response = client.get("/users/")
    assert response.status_code == 200
    # Clean up overrides after test
    app.dependency_overrides.clear()
```

## Async Tests with httpx.AsyncClient
For async endpoints requiring async database drivers:

```python
import pytest
from httpx import AsyncClient, ASGITransport

@pytest.mark.asyncio
async def test_async_root():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
```
