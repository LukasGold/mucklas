# Creating a dataclass or data model

```python
from mucklas.dataclasses import pydantic_model, model_from_signature

# Define a data model
@pydantic_model
class Person:
    name: str
    age: int
    email: str
    is_active: bool = True

# Create a data model from a function signature
def get_address(street: str, house_no: str, city: str, zip_code: str, country: str):
    return f"{street} {house_no}, {zip_code} {city} {country}"

Address = model_from_signature(get_address, "Address", strict=True)
```

# Using a data model in function calls

```python
from pydantic import validate_call
from mucklas.dataclasses import replace_param, validate_args

# Make functions that take a data model as input usable to beginners
@replace_param
def print_person(person: Person):
    print(f"{person.name} is {person.age} years old and lives in {person.email}.")

# Validate the input of a function

@validate_call
def print_x_and_y(x: str, y: int):
    print(x, y)

print_x_and_y("a", 1)  # raises a TypeError

@validate_args
def print_me(x: str, y: int):
    print(x, y)

print_me("a", "1.0")  # raises a TypeError
```
