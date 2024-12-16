from dataclasses import dataclass

import pydantic.v1 as pv1
import pytest
from pydantic import BaseModel

from mucklas.dataclasses import (
    PydanticModelVersion,
    model_from_signature,
    pydantic_model,
    pydantic_model_version_used,
    pydantic_v1_model,
    replace_args,
    replace_param,
)


def test_pydantic_model():
    @pydantic_model
    class Person:
        name: str
        age: int

    p = Person(name="Alice", age=30)
    assert p.name == "Alice"
    assert p.age == 30
    assert issubclass(Person, BaseModel)

    class Item:
        name: str
        price: float

    assert issubclass(pydantic_model(Item), BaseModel)


def test_pydantic_model_with_default():
    @pydantic_model
    class Person:
        name: str = "Alice"
        age: int = 30

    p = Person()
    assert p.name == "Alice"
    assert p.age == 30
    assert issubclass(Person, BaseModel)


def test_pydantic_v1_model():
    @pydantic_v1_model
    class Person1:
        name: str
        age: int

    p1 = Person1(name="Alice", age=30)
    assert p1.name == "Alice"
    assert p1.age == 30
    assert issubclass(Person1, pv1.BaseModel)

    class Item1:
        name: str
        price: float

    item1model = pydantic_v1_model(Item1)
    assert issubclass(item1model, pv1.BaseModel)


def test_model_from_signature():
    def func(a: int, b: float, c: str = "abc") -> str:
        return f"{a} {b} {c}"

    expected_model_name = func.__name__.capitalize() + "Param"
    model = model_from_signature(func)
    assert issubclass(model, BaseModel)
    assert model.model_fields["a"].annotation == int
    assert model.model_fields["b"].annotation == float
    assert model.model_fields["c"].annotation == str
    assert model.model_fields["c"].default == "abc"
    assert model.__name__ == expected_model_name
    globals_ = globals()
    assert expected_model_name not in globals_
    new_model_name = "MyModel"
    model1 = model_from_signature(
        func, caller_globals=globals_, model_name=new_model_name
    )
    assert model1.__name__ == new_model_name
    assert new_model_name in globals_
    with pytest.raises(ValueError):
        model_from_signature(func, caller_globals=globals_, model_name=new_model_name)

    class Param3(BaseModel):
        a: int

    class Param4(pv1.BaseModel):
        b: int

    def func1(param3: Param3, param4: Param4) -> str:
        return f"{param3.a} {param4.b}"

    with pytest.raises(ValueError):
        model_from_signature(func1)


def test_replace_param():
    class Param(BaseModel):
        a: int
        b: float
        c: str = "abc"

    @replace_param
    def func(param: Param) -> str:
        return f"{param.a} {param.b} {param.c}"

    # Test with positional arguments only
    assert func(1, 2.0) == "1 2.0 abc"
    # Test with one kwarg
    assert func(1, 2.0, c="xyz") == "1 2.0 xyz"
    # Test with multiple kwargs
    assert func(a=1, b=2.0, c="xyz") == "1 2.0 xyz"
    # Test with Param instance as positional argument
    assert func(Param(a=1, b=2.0, c="xyz")) == "1 2.0 xyz"
    # Test with Param instance as kwarg
    assert func(param=Param(a=1, b=2.0, c="xyz")) == "1 2.0 xyz"

    @dataclass
    class Param1:
        a: int
        b: float
        c: str = "abc"

    with pytest.raises(TypeError):

        @replace_param
        def func1(param: Param1) -> str:
            return f"{param.a} {param.b} {param.c}"

    with pytest.raises(ValueError):

        @replace_param
        def func2(param: Param, param1: Param1) -> str:
            return f"{param.a} {param.b} {param.c} {param1.a} {param1.b} {param1.c}"

    class Param2(BaseModel):
        a: int
        b: float
        c: str = "abc"

    class MyClass(BaseModel):
        @replace_param
        def my_method(self, param: Param2) -> str:
            return f"{param.a} {param.b} {param.c}"

    instance = MyClass()

    # Test with positional arguments only
    assert instance.my_method(1, 2.0) == "1 2.0 abc"
    # Test with one kwarg
    assert instance.my_method(1, 2.0, c="xyz") == "1 2.0 xyz"
    # Test with multiple kwargs
    assert instance.my_method(a=1, b=2.0, c="xyz") == "1 2.0 xyz"
    # Test with Param instance as positional argument
    assert instance.my_method(Param2(a=1, b=2.0, c="xyz")) == "1 2.0 xyz"
    # Test with Param instance as kwarg
    assert instance.my_method(param=Param2(a=1, b=2.0, c="xyz")) == "1 2.0 xyz"


def test_pydantic_model_version_used():
    class Person(BaseModel):
        name: str
        age: int

    class Item(pv1.BaseModel):
        name: str
        price: float

    def my_func_none(num: int, text: str):
        print(num, text)

    def my_func_comb(person: Person, item: Item):
        print(person, item)

    def my_func_v1(item: Item, item2: Item):
        print(item, item2)

    def my_func_v2(person: Person, person2: Person):
        print(person, person2)

    assert pydantic_model_version_used(my_func_none) == PydanticModelVersion.NONE
    assert pydantic_model_version_used(my_func_comb) == PydanticModelVersion.COMBINED
    assert pydantic_model_version_used(my_func_v1) == PydanticModelVersion.V1
    assert pydantic_model_version_used(my_func_v2) == PydanticModelVersion.V2


def test_replace_args():
    model_name = "ReplaceArgsModel"
    globals_ = globals()

    @replace_args(caller_globals=globals_, model_name=model_name)
    def my_func(num: int, text: str, default: str = "abc"):
        return num, text, default

    assert model_name in globals_
    # ReplaceArgsModel = globals_[model_name]  # Not necessary - just for illustration
    assert my_func(1, "text") == (1, "text", "abc")
    assert my_func(ReplaceArgsModel(num=1, text="text")) == (1, "text", "abc")  # noqa

    model_name = "DescribeMeModel"

    class Person(BaseModel):
        name: str
        age: int

        @replace_args(caller_globals=globals_, model_name=model_name)
        def describe_me(self, verbose: bool = False):
            if verbose:
                return f"{self.name} is {self.age} years old."
            else:
                return self.name

    assert model_name in globals_
    assert Person(name="Alice", age=30).describe_me() == "Alice"
    assert (
        Person(name="Alice", age=30).describe_me(verbose=True)
        == "Alice is 30 years old."
    )
    # print(Person(name="Alice", age=30).describe_me(DescribeMeModel(verbose=False)))
    assert (
        Person(name="Tom", age=20).describe_me(DescribeMeModel(verbose=False))  # noqa
        == "Tom"
    )
    assert (
        Person(name="Tom", age=20).describe_me(DescribeMeModel(verbose=True))  # noqa
        == "Tom is 20 years old."
    )

    class Child(Person):
        parent: Person

        @classmethod
        @replace_args(caller_globals=globals_)
        def return_class_name(cls, verbose: bool = False):
            if verbose:
                return f"Class name: {cls.__name__}"
            return cls.__name__

    assert (
        Child(
            name="Anna", age=3, parent=Person(name="Alice", age=25)
        ).return_class_name()
        == "Child"
    )
    assert (
        Child.return_class_name(ReturnClassNameParam(verbose=True))  # noqa
        == "Class name: Child"
    )

    # Not implemented yet:
    # with pytest.raises(ValueError):
    #     # Mixing of different model versions is not allowed
    #
    #     class Package(pv1.BaseModel):
    #         id: int
    #
    #         @replace_args(caller_globals=globals_)
    #         def return_me(self, to: Person, verbose: bool = False):
    #             if verbose:
    #                 return f"Returning package {self.id} to {to.name}"
    #             else:
    #                 return f"{self.id} back to {to.name}"
