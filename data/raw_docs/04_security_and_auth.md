# FastAPI: Security and Authentication

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
