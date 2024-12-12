import inspect
from dataclasses import dataclass
from enum import StrEnum
from functools import partialmethod, wraps
from typing import Any, Callable, Dict, Type, TypeVar, Union, get_args

import pydantic.v1 as pv1
from pydantic import BaseModel, ConfigDict, create_model
from typing_extensions import deprecated

T = TypeVar("T")


def partialclass(cls, *args, **kwargs):  # noqa: typo
    """Partial class to be used on a class to partially initialize it.
    E.g., to set default values for some attributes. Returns a partially initialized
    class, which can be used to create instances of the class, but without having to
    set the default values for the attributes.

    Source
    ------
    https://stackoverflow.com/questions/38911146/
    python-equivalent-of-functools-partial-for-a-class-constructor
    """

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def pydantic_model(cls: Type[T]) -> Type[T]:
    """Decorator to be used on a class definition to create a Pydantic (v2) model
    from a dataclass."""
    return create_model(cls.__name__, __base__=(BaseModel, cls), __config__=None)


def pydantic_v1_model(cls: Type[T]) -> Type[T]:
    """Decorator to be used on a class definition to create a Pydantic (v1) model
    from a dataclass."""
    # Create a dictionary of fields from the class annotations
    fields = {key: (value, ...) for key, value in cls.__annotations__.items()}
    return pv1.create_model(cls.__name__, __base__=(pv1.BaseModel,), **fields)


class PydanticModelVersion(StrEnum):
    V1 = "v1"
    V2 = "v2"
    NONE = "none"
    OTHER = "other"
    COMBINED = "combined"


# Use dataclass to not mix BaseModel and pv1.BaseModel
@dataclass
class WhatModelTypeResult:
    pydantic_model_version: PydanticModelVersion
    model: Union[Type[BaseModel], Type[pv1.BaseModel]] = None


def what_model_type(type_: Type[T], return_model: bool = False) -> WhatModelTypeResult:
    """Determines if a type is a (subclass of a) Pydantic model and if so,
    which version of Pydantic model is used.

    Parameters
    ----------
    type_
        The type to be tested.
    return_model
        Whether to return the model if the type is a Pydantic model.

    Returns
    -------
    result
        The Pydantic model version used and the model if return_model is True.
    """
    result = PydanticModelVersion.NONE
    model: Union[pv1.BaseModel, BaseModel, None] = None
    if (
        getattr(type_, "__origin__", None) is not None
        and getattr(type_, "__origin__", None) == Union
    ):
        # Test if it is a Union with a BaseModel
        # Go through the arguments of the Union
        for arg in get_args(type_):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                result = PydanticModelVersion.V2
                model: Type[BaseModel] = arg
                break
            elif isinstance(arg, type) and issubclass(arg, pv1.BaseModel):
                model: Type[pv1.BaseModel] = arg
                result = PydanticModelVersion.V1
                break
    elif issubclass(type_, BaseModel):
        result = PydanticModelVersion.V2
        model: Type[BaseModel] = type_
    elif issubclass(type_, pv1.BaseModel):
        result = PydanticModelVersion.V1
        model: Type[pv1.BaseModel] = type_
    if return_model:
        return WhatModelTypeResult(pydantic_model_version=result, model=model)
    return WhatModelTypeResult(pydantic_model_version=result)


def pydantic_model_version_used(func: Callable) -> PydanticModelVersion:
    """Determines which version of Pydantic models is used in the type hints of a
    function signature. If the function signature contains a mix of Pydantic model
    versions, it is reported as 'combined'.

    Parameters
    ----------
    func
        The function to be tested

    Returns
    -------
    result
        The Pydantic model version used in the function signature
    """
    signature = inspect.signature(func)
    # Get the argument types
    arg_types = {
        name: what_model_type(param.annotation).pydantic_model_version
        for name, param in signature.parameters.items()
    }
    # Get the unique types
    unique_types = set(arg_types.values())
    # Test if all arguments are of the same type
    if (
        PydanticModelVersion.V1 in unique_types
        and PydanticModelVersion.V2 in unique_types
    ):
        return PydanticModelVersion.COMBINED
    elif PydanticModelVersion.V1 in unique_types:
        return PydanticModelVersion.V1
    elif PydanticModelVersion.V2 in unique_types:
        return PydanticModelVersion.V2
    return PydanticModelVersion.NONE


def model_from_signature(
    func: Callable,
    caller_globals: dict = None,
    model_name: str = None,
    model_description: str = None,
    strict: bool = False,
) -> Type[BaseModel]:
    """Function to create a Pydantic model from a function signature, using type
    hints to annotate the model fields.

    Parameters
    ----------
    func
        The function from which to extract the type hints.
    caller_globals
        The globals of the calling module. Pass `globals()` to use the globals of the
        calling module. If provided, the model will be added to the globals of the
        calling module. Otherwise, the new model will only be returned.
    model_name
        The name of the Pydantic model to be created. If not provided, the name will be
        generated based on the function name.
    model_description
        The description of the Pydantic model. If not provided, a default description
        will be generated.
    strict
        Whether to set the model to strict validation. Default is False.

    Returns
    -------
    result
        The Pydantic model created from the function signature.
    """
    # Extract the function signature
    signature = inspect.signature(func)
    # Test if the functions arguments mix BaseModel and pv1.BaseModel
    model_version = pydantic_model_version_used(func)
    if model_version == PydanticModelVersion.COMBINED:
        raise ValueError(
            "The function signature must have type hints that are all of the same "
            "type, either Pydantic BaseModel or Pydantic v1 BaseModel."
        )
    # Create a dictionary with the field names and their types
    model_fields: Dict[str, Any] = {
        name: (
            param.annotation,
            param.default if param.default is not inspect.Parameter.empty else ...,
        )
        for name, param in signature.parameters.items()
    }
    # Create a model name if not provided
    if model_name is None:
        model_name = (
            "".join([s.lower().capitalize() for s in func.__name__.split("_")])
            + "Param"
        )
    # Create a model description if not provided
    if model_description is None:
        model_description = (
            f"Pydantic model created from the type hints in "
            f"the signature of function{func.__name__}'"
        )
    # Set the model description
    model_fields["__doc__"] = model_description
    # Create a Pydantic model dynamically, based on the model version
    if model_version == PydanticModelVersion.V1:
        # Set validation to strict
        if strict is not False:
            model_fields["Config"] = pv1.ConfigDict(
                strict=strict, arbitrary_types_allowed=True
            )
        new_model: Type[pv1.BaseModel] = pv1.create_model(model_name, **model_fields)
    else:
        # elif model_version == PydanticModelVersion.V2:
        # Set validation to strict
        if strict is not False:
            model_fields["model_config"] = ConfigDict(
                strict=strict, arbitrary_types_allowed=True
            )
        new_model: Type[BaseModel] = create_model(model_name, **model_fields)
    if caller_globals is not None:
        if model_name in caller_globals:
            raise ValueError(
                f"A variable with the name '{model_name}' already exists in the "
                "globals of the calling module. Please provide a different model name."
            )
        # Add the model to the globals of the calling module
        caller_globals[model_name] = new_model
    return new_model


def replace_param(func: Callable) -> Callable:
    """Decorator that will modify the function signature to accept keyword arguments
    corresponding to the fields of a Pydantic model. The Pydantic model is expected
    to be the only parameter of the function.

    Parameters
    ----------
    func
        The function to be decorated

    Returns
    -------
    wrapper
        The decorated function
    """
    # Get the original signature
    original_sig: inspect.Signature = inspect.signature(func)

    # See if the function has a single parameter
    if len(original_sig.parameters) != 1:
        raise ValueError(
            f"The function '{func.__name__}' must have exactly one parameter of type "
            f"Pydantic model for the decorator to work properly."
        )
    # Get the name of the parameter
    param_name: str = list(original_sig.parameters.keys())[0]
    # Extract the params argument type
    param_type = original_sig.parameters[param_name].annotation
    # Ensure params_type is a subclass of BaseModel
    test_res = what_model_type(type_=param_type, return_model=True)
    if test_res.pydantic_model_version == PydanticModelVersion.NONE:
        raise TypeError(
            f"The type hint of the single parameter of the function '{func.__name__}' "
            "must be a subclass of Pydantic BaseModel. If the type hint is a Union, "
            "one argument of Union must be a subclass of Pydantic BaseModel."
        )
    model: Union[Type[BaseModel], Type[pv1.BaseModel]] = test_res.model
    # test_res.model
    # Create new parameters based on the fields of the params_type
    if test_res.pydantic_model_version == PydanticModelVersion.V1:
        model_: Type[pv1.BaseModel] = model
        new_params = [
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=field.default
            )
            for name, field in model_.__fields__.items()
        ]
    else:
        model_: Type[BaseModel] = model
        new_params = [
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=field.default
            )
            for name, field in model_.model_fields.items()
        ]

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Unfortunately, it is not possible to programmatically overload the decorated
        function with a signature that contains kwargs and at the same time keep the
        signature with the model instance as the only argument."""
        if len(args) == 1 and isinstance(args[0], model):
            # There is only one positional argument and it is an instance of the model
            return func(args[0])
        elif len(args) == 0 and isinstance(kwargs.get(param_name), model):
            # There are no positional arguments and the only keyword argument is an
            #  instance of the model
            return func(kwargs.get(param_name))
        else:
            if len(args) == 0:
                # There are no positional arguments
                # Create an instance of the model with the provided keyword arguments
                params_instance = model(**kwargs)
            else:
                # There are positional and keyword arguments
                # Pydantic does not allow positional arguments, so we need to map the
                #  positional arguments to the field names of the model
                # Get the name of potential arguments
                if test_res.pydantic_model_version == PydanticModelVersion.V1:
                    model__: Type[pv1.BaseModel] = model
                    pot_args = model__.__fields__.keys()
                else:
                    model__: Type[BaseModel] = model
                    pot_args = model__.model_fields.keys()
                # Create a dictionary with the positional arguments mapped to the fields
                new_args = dict(zip(pot_args, args))
                # Merge the positional arguments with the keyword arguments
                #  and create an instance of the model
                params_instance = model(**{**new_args, **kwargs})
            # Call the original function with the model instance as the only argument
            return func(params_instance)

    # Update the signature of the wrapper function with the new parameters
    wrapper.__signature__ = original_sig.replace(parameters=new_params)
    return wrapper


@deprecated("Use pydantic.validate_call instead.")
def validate_args(func: Callable) -> Callable:
    """Use a pydantic model to validate if the args and kwargs are correctly typed."""
    # Create a Pydantic model from the function signature
    model = model_from_signature(func, strict=True)

    # Get the args and kwargs that were actually used when calling the function
    def wrapper(*args, **kwargs):
        # Get the names of the positional args from the function signature
        arg_names = list(inspect.signature(func).parameters.keys())
        pos_args: Dict[str, Any] = dict(zip(arg_names, args))
        # Try to initiate the model with the args and kwargs
        try:
            _ = model(**{**pos_args, **kwargs})
        except Exception as e:
            raise ValueError(
                f"Error when validating the arguments of function '{func.__name__}':\n"
                f"{e}"
            )
        return func(*args, **kwargs)

    return wrapper


def replace_args(caller_globals: dict, model_name: str = None) -> Callable:
    """Decorator that will modify the function signature to accept a single
    parameter, which will be a Pydantic model with annotations corresponding to
    the keyword arguments of the function.

    Parameters
    ----------
    caller_globals
        Required - The globals of the calling module. Pass `globals()` to use the
        globals of the calling module.
    model_name
        The name of the Pydantic model to be created. If not provided, the name will be
        generated based on the function name.

    Returns
    -------
    wrapper
        The decorated function
    """
    # todo: what if this decorator is used on a method?

    def decorator(func: Callable):
        """Inner decorator to decorate the function

        Parameters
        ----------
        func
            The function to be decorated.
        """
        # Create a Pydantic model from the function signature
        model = model_from_signature(func, caller_globals, model_name)
        model_version = what_model_type(model).pydantic_model_version
        if model_version == PydanticModelVersion.V1:
            method_name = "dict"
        else:
            method_name = "model_dump"
        # Create a new signature with the model as the only parameter
        original_sig = inspect.signature(func)
        new_sig = original_sig.replace(
            parameters=[
                inspect.Parameter(
                    "params",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=model,
                    default=inspect.Parameter.empty,
                )
            ]
        )

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) == 1 and isinstance(args[0], model):
                method = getattr(args[0], method_name)
                return func(**method())
            elif (
                len(args) == 0
                and len(kwargs) == 0
                and isinstance(kwargs.get("params"), model)
            ):
                method = getattr(kwargs["params"], method_name)
                return func(**method())
            else:
                return func(*args, **kwargs)

        wrapper.__signature__ = new_sig
        return wrapper

    return decorator


if __name__ == "__main__":

    class Person(BaseModel):
        name: str
        age: int

    class Item(pv1.BaseModel):
        name: str
        price: float

    # @replace_param
    # def my_func(person: Person):
    #     print(person)
    #
    # instance = Person(name="Alice", age=30)
    # signature = inspect.signature(my_func)
    # print(signature)
    #
    # @replace_args
    # def my_com_func(x: str, y: int, z: Person, a: float = 1.0):
    #     print(x, y, z, a)
    #
    # # Model = model_from_signature(my_com_func)
    #
    # # mm = Model()
