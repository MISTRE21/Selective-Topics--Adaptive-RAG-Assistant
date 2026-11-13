# FastAPI: Background Tasks

## What are Background Tasks?
Background tasks allow long-running operations (like sending welcome emails, processing uploaded logs, or updating analytics) to execute after returning an HTTP response to the client.

## Using `BackgroundTasks`
Import `BackgroundTasks` from `fastapi` and declare it as a parameter in your route function:

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def send_notification(email: str, message: str = ""):
    with open("notifications.log", mode="a") as f:
        f.write(f"Notification to {email}: {message}
")

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
