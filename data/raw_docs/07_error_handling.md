# FastAPI: Error Handling and Exceptions

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
