# FastAPI: Middleware and CORS

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
