import json
import os

DOCS = {
    "01_routing_and_parameters.md": """# FastAPI: Routing and Parameters

## Overview
FastAPI allows declaring path parameters, query parameters, request bodies, and custom status codes using standard Python type annotations. FastAPI uses these declarations for automatic data parsing, validation, serialization, and interactive OpenAPI documentation.

## Path Parameters
You declare path "parameters" or "variables" with standard Python format string syntax `{param_name}`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The type hint `item_id: int` enforces integer validation. If a user passes `/items/foo`, FastAPI returns an HTTP 422 Unprocessable Entity error with detailed error metadata.

### Path Parameters with Predefined Values
If you have a path parameter that receives predefined values, you can use a Python `Enum`:

```python
from enum import Enum
from fastapi import FastAPI

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    return {"model_name": model_name, "message": "Have some residuals"}
```

## Query Parameters
Function parameters that are not part of path parameters are automatically interpreted as query parameters:

```python
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10, q: str | None = None):
    return {"skip": skip, "limit": limit, "q": q}
```

Default values make the parameter optional. When `q: str | None = None`, omitting `q` sets it to `None`.

## Request Body and Status Codes
To declare a request body, combine Pydantic models with the HTTP decorator. You can specify response status codes via `status_code` argument in decorators:

```python
from fastapi import FastAPI, status
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = False

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item
```

## Response Model and Filtering
Use `response_model` to filter sensitive data from output responses:

```python
@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn):
    return user  # UserOut filters out password_hash automatically
```
""",

    "02_request_validation_pydantic.md": """# FastAPI: Request Validation and Pydantic

## Pydantic Models
FastAPI uses Pydantic to define schemas and validate incoming request payloads. When an endpoint receives data, Pydantic parses the payload into declared types, validates constraints, and raises `RequestValidationError` if validation fails.

```python
from pydantic import BaseModel, Field, EmailStr

class UserProfile(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Unique handle")
    email: EmailStr
    age: int = Field(gt=0, le=120, description="Age in years")
    bio: str | None = Field(default=None, max_length=300)
```

## Nested Models and Lists
Pydantic models can be deeply nested to validate complex JSON trees:

```python
from pydantic import BaseModel

class Tag(BaseModel):
    name: str
    category: str

class Article(BaseModel):
    title: str
    content: str
    tags: list[Tag] = []
```

## Field Validation and Custom Validators
In Pydantic v2 (supported by modern FastAPI), you use `@field_validator` and `@model_validator`:

```python
from pydantic import BaseModel, field_validator

class Account(BaseModel):
    pin: str

    @field_validator("pin")
    @classmethod
    def validate_pin(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 4:
            raise ValueError("PIN must be exactly 4 digits")
        return v
```

## Query and Path Validation via FastAPI
FastAPI also provides `Query`, `Path`, `Header`, and `Cookie` helpers for scalar validation:

```python
from fastapi import FastAPI, Query, Path

app = FastAPI()

@app.get("/search/")
async def search(
    q: str = Query(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_]+$"),
    page: int = Query(1, ge=1)
):
    return {"query": q, "page": page}
```
""",

    "03_dependency_injection.md": """# FastAPI: Dependency Injection

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
""",

    "04_security_and_auth.md": """# FastAPI: Security and Authentication

## OAuth2 with Password Flow and Bearer Tokens
FastAPI provides built-in utilities in `fastapi.security` for OAuth2, API Keys, and HTTP Basic/Bearer auth.

```python
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(token: str = Depends(oauth2_scheme)):
    user = decode_token(token)
    return user
```

## Security Scopes
OAuth2 scopes allow granular permissions:

```python
from fastapi.security import SecurityScopes

async def get_current_user(security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)):
    if security_scopes.scopes:
        authenticate_scopes(token, security_scopes.scopes)
    return decode_user(token)
```

## HTTPBearer and APIKeyHeader
For simple token authorization without OAuth2 password flow:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

auth_scheme = HTTPBearer()

@app.get("/api/protected")
def protected_route(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    return {"token": credentials.credentials}
```
""",

    "05_middleware_and_cors.md": """# FastAPI: Middleware and CORS

## What is Middleware?
Middleware is a function that runs before every request is processed by an endpoint, and after every response is generated. It can inspect and modify request headers, enforce rate limits, calculate execution timing, or block requests.

## Custom Middleware with `http`
Use `@app.middleware("http")` to declare HTTP middleware:

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

`call_next` passes the request to the endpoint or subsequent middleware.

## CORS (Cross-Origin Resource Sharing)
To allow browsers to make requests from different origins (domains/ports), add `CORSMiddleware`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## TrustedHostMiddleware and GZipMiddleware
FastAPI also ships with standard Starlette middleware:

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["example.com", "*.example.com"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
```
""",

    "06_background_tasks.md": """# FastAPI: Background Tasks

## What are Background Tasks?
Background tasks allow long-running operations (like sending welcome emails, processing uploaded logs, or updating analytics) to execute after returning an HTTP response to the client.

## Using `BackgroundTasks`
Import `BackgroundTasks` from `fastapi` and declare it as a parameter in your route function:

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def send_notification(email: str, message: str = ""):
    with open("notifications.log", mode="a") as f:
        f.write(f"Notification to {email}: {message}\n")

@app.post("/send-notification/{email}")
async def send_notification_endpoint(
    email: str,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(send_notification, email, message="Welcome to the service!")
    return {"message": "Notification scheduled in the background"}
```

## Background Tasks in Dependencies
You can also inject `BackgroundTasks` inside dependencies:

```python
def write_audit_log(event: str):
    log_to_db(event)

def get_current_user_with_audit(
    background_tasks: BackgroundTasks,
    token: str = Depends(oauth2_scheme)
):
    user = verify_token(token)
    background_tasks.add_task(write_audit_log, f"User {user.id} accessed route")
    return user
```

## Difference from Celery / Task Queues
`BackgroundTasks` runs inside the same asyncio event loop or threadpool as the FastAPI server process. For heavy distributed computing, CPU-intensive image rendering, or worker persistence across reboots, use Celery, RQ, or ARQ instead.
""",

    "07_error_handling.md": """# FastAPI: Error Handling and Exceptions

## HTTPException
To return HTTP error responses to the client, raise `HTTPException`:

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
            headers={"X-Error": "ResourceMissing"}
        )
    return {"item": items[item_id]}
```

## Custom Exception Handlers
You can create custom exception classes and register global handlers with `@app.exception_handler()`:

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

app = FastAPI()

@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something weird. Rainbows!"},
    )
```

## Overriding Request Validation Errors
To customize the default 422 JSON response when request validation fails:

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse(str(exc), status_code=400)
```
""",

    "08_testing_and_overrides.md": """# FastAPI: Testing and Dependency Overrides

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
"""
}

# Write raw documentation files
out_dir = r"C:\Users\mistr\.gemini\antigravity\scratch\adaptive_rag_assistant\data\raw_docs"
os.makedirs(out_dir, exist_ok=True)
for fname, content in DOCS.items():
    p = os.path.join(out_dir, fname)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
print(f"Successfully generated {len(DOCS)} documentation files in {out_dir}")
