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


# Partial definition of a data model

```python
from osw.express import OswExpress
from mucklas.dataclasses import partialclass

# OswExpress.StoreEntityParam
'''
entities: Union[OswBaseModel, List[OswBaseModel]]  # actually model.Entity
"""The entities to store. Can be a single entity or a list of entities."""
namespace: Optional[str]
"""The namespace of the entities. If not set, the namespace is derived from the
entity."""
parallel: Optional[bool] = None
"""If set to True, the entities are stored in parallel."""
overwrite: Optional[OVERWRITE_CLASS_OPTIONS] = "keep existing"
"""If no class specific overwrite setting is set, this setting is used."""
overwrite_per_class: Optional[List[OSW.OverwriteClassParam]] = None
"""A list of OverwriteClassParam objects. If a class specific overwrite setting
is set, this setting is used.
"""
meta_category_title: Optional[str] = "Category:Category"
debug: Optional[bool] = False
_overwrite_per_class: Dict[str, Dict[str, OSW.OverwriteClassParam]] = (
    PrivateAttr()
)
'''

# Replacing the default value for "parallel"
ParallelDisabled = partialclass(parallel=False)(OswExpress.StoreEntityParam)

osw_obj = OswExpress("wiki-dev.open-semantic-lab.org")

osw_obj.store_entity(ParallelDisabled(entities=[]))
```
