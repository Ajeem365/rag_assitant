# FastAPI — Official Documentation Excerpt

## What is FastAPI?

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints. It is one of the fastest Python frameworks available, on par with NodeJS and Go.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go
- **Fast to code**: Increase the speed to develop features by about 200% to 300%
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors
- **Intuitive**: Great editor support. Completion everywhere. Less time debugging
- **Easy**: Designed to be easy to use and learn. Less time reading docs
- **Short**: Minimize code duplication. Multiple features from each parameter declaration
- **Robust**: Get production-ready code with automatic interactive documentation
- **Standards-based**: Based on (and fully compatible with) the open standards for APIs: OpenAPI and JSON Schema

---

## Installation

```bash
pip install fastapi
pip install "uvicorn[standard]"
```

---

## First Steps — Hello World

Create a file `main.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run it:

```bash
uvicorn main:app --reload
```

Open your browser at `http://127.0.0.1:8000`. You will see the JSON response: `{"message": "Hello World"}`.

---

## Path Parameters

You can declare path "parameters" or "variables" with the same syntax used by Python format strings:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
```

The value of the path parameter `item_id` will be passed to your function as the argument `item_id`. FastAPI will automatically validate and convert the type — if you pass a string where an int is expected, it returns a clear HTTP 422 error.

### Order Matters

When creating path operations, you can find situations where you have a fixed path. Like `/users/me` — you want it to return data about the current user. And you also have the path `/users/{user_id}` — to get data about a specific user by their ID. Because path operations are evaluated in order, you need to make sure that the path for `/users/me` is declared before the one for `/users/{user_id}`.

---

## Query Parameters

When you declare other function parameters that are not part of the path parameters, they are automatically interpreted as "query" parameters:

```python
from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]
```

The query is the set of key-value pairs that go after the `?` in a URL, separated by `&` characters. For example: `http://127.0.0.1:8000/items/?skip=0&limit=10`

### Optional Query Parameters

```python
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Optional[str] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
```

---

## Request Body

When you need to send data from a client to your API, you send it as a request body. Use Pydantic models:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

@app.post("/items/")
async def create_item(item: Item):
    return item
```

---

## Response Model

You can declare the model used for the response with the `response_model` parameter:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

class ItemOut(BaseModel):
    name: str

@app.post("/items/", response_model=ItemOut)
async def create_item(item: Item):
    return item
```

FastAPI will filter the output data to match the `response_model`, even if it has extra fields.

---

## HTTP Status Codes

```python
from fastapi import FastAPI, status

app = FastAPI()

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(name: str):
    return {"name": name}
```

---

## Dependency Injection

FastAPI has a very powerful but intuitive Dependency Injection system:

```python
from fastapi import Depends, FastAPI

app = FastAPI()

async def common_parameters(q: str | None = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

---

## Error Handling

Raise HTTP exceptions with `HTTPException`:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

items = {"foo": "The Foo Wrestlers"}

@app.get("/items/{item_id}")
async def read_item(item_id: str):
    if item_id not in items:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items[item_id]}
```

---

## Background Tasks

```python
from fastapi import BackgroundTasks, FastAPI

app = FastAPI()

def write_notification(email: str, message: str = ""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@app.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

---

## Middleware

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
