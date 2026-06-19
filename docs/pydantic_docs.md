# Pydantic v2 — Data Validation Documentation

## What is Pydantic?

Pydantic is the most widely used data validation library for Python. It uses Python type annotations to validate data at runtime, providing clear error messages when data doesn't match the expected schema.

---

## Basic Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    id: int
    name: str
    email: str
    age: Optional[int] = None
    is_active: bool = True

# Creating an instance — validates automatically
user = User(id=1, name="Zain", email="zain@example.com", age=22)
print(user.model_dump())
# {'id': 1, 'name': 'Zain', 'email': 'zain@example.com', 'age': 22, 'is_active': True}
```

---

## Field Validation

Use `Field()` for advanced constraints:

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0, description="Price in USD")
    quantity: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)
```

---

## Validators

```python
from pydantic import BaseModel, field_validator, model_validator

class Order(BaseModel):
    item: str
    quantity: int
    price: float
    total: float = 0.0

    @field_validator("quantity")
    @classmethod
    def quantity_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Quantity must be positive")
        return v

    @model_validator(mode="after")
    def compute_total(self):
        self.total = self.quantity * self.price
        return self
```

---

## Serialization

```python
user = User(id=1, name="Zain", email="z@e.com")

# To dict
user.model_dump()

# To JSON string
user.model_dump_json()

# Exclude None values
user.model_dump(exclude_none=True)

# Include only specific fields
user.model_dump(include={"id", "name"})
```

---

## Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str = "India"

class Person(BaseModel):
    name: str
    address: Address

p = Person(name="Zain", address={"street": "123 Main St", "city": "Bareilly"})
print(p.address.city)  # Bareilly
```

---

## Settings Management

```python
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False

    model_config = {"env_file": ".env"}

settings = AppSettings()  # Reads from environment / .env
```

---

## Error Handling

```python
from pydantic import ValidationError

try:
    user = User(id="not-an-int", name="Zain", email="z@e.com")
except ValidationError as e:
    print(e.json())
    # Detailed error with field, message, type
```
