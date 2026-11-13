# FastAPI: Dependency Injection

## What is Dependency Injection?
Dependency Injection (DI) allows endpoints to declare dependencies required for business logic, database sessions, authentication, or external API clients. FastAPI resolves dependencies hierarchically and caches results across a single request when configured.

## Defining and Using Dependencies
Dependencies are declared using `Depends`:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_pagination(skip: int = 0, limit: int = 20):
    return {"skip": skip, "limit": limit}

@app.get("/users/")
async def get_users(pagination: dict = Depends(common_pagination)):
    return {"users": ["alice", "bob"], "pagination": pagination}
```

## Class-based Dependencies
Classes can also be used as dependencies:

```python
class CommonQueryParams:
    def __init__(self, q: str | None = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(params: CommonQueryParams = Depends(CommonQueryParams)):
    return {"params": params.__dict__}
```

## Dependencies with `yield` (Context Managers / DB Sessions)
FastAPI supports dependencies that use `yield` instead of `return`. This allows code before `yield` to run as setup, and code after `yield` to run as teardown even if exceptions occur during request handling:

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
```

## Global and Router-level Dependencies
You can apply dependencies across an entire `APIRouter` or `FastAPI` instance:

```python
from fastapi import FastAPI, Depends

app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])
```
