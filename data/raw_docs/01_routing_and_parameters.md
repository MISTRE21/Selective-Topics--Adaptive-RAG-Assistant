# FastAPI: Routing and Parameters

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
