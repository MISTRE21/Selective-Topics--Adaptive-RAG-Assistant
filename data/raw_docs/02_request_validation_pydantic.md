# FastAPI: Request Validation and Pydantic

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
