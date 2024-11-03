from pydantic import BaseModel

from mucklas.dataclasses import model_from_signature, pydantic_model, replace_param


def test_pydantic_model():
    @pydantic_model
    class Person:
        name: str
        age: int

    p = Person(name="Alice", age=30)
    assert p.name == "Alice"
    assert p.age == 30
    assert issubclass(Person, BaseModel)


def test_pydantic_model_with_default():
    @pydantic_model
    class Person:
        name: str = "Alice"
        age: int = 30

    p = Person()
    assert p.name == "Alice"
    assert p.age == 30
    assert issubclass(Person, BaseModel)


def test_model_from_signature():
    def func(a: int, b: float, c: str = "abc") -> None:
        pass

    model = model_from_signature(func)
    assert issubclass(model, BaseModel)
    assert model.model_fields["a"].annotation == int
    assert model.model_fields["b"].annotation == float
    assert model.model_fields["c"].annotation == str
    assert model.model_fields["c"].default == "abc"


def test_replace_param():
    class Param(BaseModel):
        a: int
        b: float
        c: str = "abc"

    @replace_param
    def func(param: Param) -> str:
        return f"{param.a} {param.b} {param.c}"

    assert func(a=1, b=2.0, c="xyz") == "1 2.0 xyz"
    assert func(a=1, b=2.0) == "1 2.0 abc"
