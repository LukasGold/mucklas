import inspect
import sys
from dataclasses import dataclass

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from backports.strenum import StrEnum

import functools
from functools import partialmethod, wraps
from typing import Any, Callable, Dict, Type, TypeVar, Union, get_args

import pydantic.v1 as pv1
from pydantic import BaseModel, ConfigDict, create_model
from typing_extensions import deprecated

T = TypeVar("T")


class FuncType(StrEnum):
    function = "function"
    method = "method"
    class_method = "class_method"


def partialclass(*args, **kwargs) -> Callable[[type], type]:  # noqa: typo
    """Partial class to be used on a class to partially initialize it.
    E.g., to set default values for some attributes. Returns a partially initialized
    class, which can be used to create instances of the class, but without having to
    set the default values for the attributes.

    Source
    ------
    https://stackoverflow.com/questions/38911146/
    python-equivalent-of-functools-partial-for-a-class-constructor
    """

    def decorator(cls: type) -> type:
        class NewCls(cls):
            __init__ = partialmethod(cls.__init__, *args, **kwargs)

        functools.update_wrapper(NewCls, cls, updated=())
        return NewCls

    return decorator


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


def what_model_version(
    type_: Union[type, Type[T]], return_model: bool = False
) -> WhatModelTypeResult:
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


def attribute_defaults(mdl: Union[Type[pv1.BaseModel], Type[BaseModel]]):
    """Function to extract the default values of the fields of a Pydantic model."""
    model_version = what_model_version(mdl).pydantic_model_version
    defaults = {}
    if model_version == PydanticModelVersion.V1:
        mdl: Type[pv1.BaseModel]
        model_fields = mdl.__fields__
    elif model_version == PydanticModelVersion.V2:
        mdl: Type[BaseModel]
        model_fields = mdl.model_fields
    else:
        raise ValueError("The model must be a Pydantic model.")
    for name, field in model_fields.items():
        defaults[name] = (
            field.default
            if field.default is not None
            else (field.default_factory if field.default_factory is not None else None)
        )
    return defaults


def function_or_method(func: Callable) -> FuncType:
    """Determines if a function is a standalone function, a method of a class, or a
    class method.

    Parameters
    ----------
    func
        The function to be tested

    Returns
    -------
    result
        The type of the function
    """
    original_sig: inspect.Signature = inspect.signature(func)
    func_type = FuncType.function
    if len(original_sig.parameters) > 0:
        # print("The function has more than zero parameters")
        first_param = list(original_sig.parameters.values())[0]
        # print("Name of first parameter:", first_param.name)
        # print("ismethod:", inspect.ismethod(func))
        if first_param.name == "self":
            # print("The function is a method of class")
            func_type = FuncType.method
        elif first_param.name == "cls":
            # print("The function is a class method")
            func_type = FuncType.class_method
    return func_type


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
        name: what_model_version(param.annotation).pydantic_model_version
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
    model_version: PydanticModelVersion = None,
    strict: bool = False,
    arbitrary_types_allowed: bool = True,
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
    model_version
        The version of Pydantic model to be created. If not provided, the version will
        be determined based on the type hints in the function signature.
    strict
        Whether to set the model to strict validation. Default is False.
    arbitrary_types_allowed
        Whether to allow arbitrary types in the model. Default is True.

    Returns
    -------
    result
        The Pydantic model created from the function signature.
    """
    # Extract the function signature
    signature = inspect.signature(func)
    if len(signature.parameters) == 0:
        raise ValueError(
            "The passed function must have at least one parameter to"
            " create a Pydantic model."
        )
    # Test if the functions arguments mix BaseModel and pv1.BaseModel
    if model_version is None:
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
        if name != "self" and name != "cls"
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
            f"the signature of function {func.__name__}'"
        )
    # Set the model description
    model_fields["__doc__"] = model_description
    # Create a Pydantic model dynamically, based on the model version
    if model_version == PydanticModelVersion.V1:
        # Set validation to strict
        if strict is not False:
            model_fields["Config"] = pv1.ConfigDict(
                strict=strict, arbitrary_types_allowed=arbitrary_types_allowed
            )
        new_model: Type[pv1.BaseModel] = pv1.create_model(model_name, **model_fields)
    else:
        # elif model_version == PydanticModelVersion.V2:
        # Set validation to strict
        if strict is not False:
            model_fields["model_config"] = ConfigDict(
                strict=strict, arbitrary_types_allowed=arbitrary_types_allowed
            )
        print(model_fields)
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


def compare_param_model_version(func: Callable, model_version: PydanticModelVersion):
    """Compares the Pydantic model version used in the function signature of a method
    and the passed Pydantic model version.

    Parameters
    ----------
    func
        The method to be tested
    model_version
        The Pydantic model version to be compared with

    Raises
    ------
    ValueError
        If the Pydantic model version used in the function signature of the method is
        different from the Pydantic model version used in the class that posses the
        method.
    """
    original_sig: inspect.Signature = inspect.signature(func)
    func_type = function_or_method(func)
    class_model_version = PydanticModelVersion.NONE
    if func_type == FuncType.method:
        # Not implemented yet. Approaches tried so far:
        # - Access signature and get the class from the annotation of the first
        #   parameter
        # - Derive the class name from the qualified name of the method
        # - Access the __self__ attribute of the method
        pass
    elif func_type == FuncType.class_method:
        pass
    # Compare the Pydantic model version used in the function signature of the method
    if (
        class_model_version != PydanticModelVersion.NONE
        and class_model_version != model_version
    ):
        raise ValueError(
            f"The Pydantic model version used in the function signature of "
            f"'{func.__name__}' is different from the Pydantic model version "
            f"used in the class "
            f"'{original_sig.parameters['self'].annotation.__class__.__name__}'."
        )


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
    # Test if the function is a method of a class
    func_type = function_or_method(func)
    # See if the function has a single parameter (except for self or cls)
    if func_type == FuncType.function and len(original_sig.parameters) != 1:
        raise ValueError(
            f"The function '{func.__name__}' must have exactly one parameter of type "
            f"Pydantic model for the decorator to work properly."
        )
    elif func_type != FuncType.function and len(original_sig.parameters) != 2:
        raise ValueError(
            f"The function '{func.__name__}' must have exactly one parameter of type "
            f"Pydantic model other than 'self' or 'cls' for the decorator to work "
            f"properly."
        )
    # Get the name of the parameter
    if func_type == FuncType.function:
        param_name: str = list(original_sig.parameters.keys())[0]
    else:
        param_name: str = list(original_sig.parameters.keys())[1]
    # Extract the params argument type
    param_type = original_sig.parameters[param_name].annotation
    # Ensure params_type is a subclass of BaseModel
    test_res = what_model_version(type_=param_type, return_model=True)
    model_version = test_res.pydantic_model_version
    if model_version == PydanticModelVersion.NONE:
        raise TypeError(
            f"The type hint of the single parameter of the function '{func.__name__}' "
            "must be a subclass of Pydantic BaseModel. If the type hint is a Union, "
            "one argument of Union must be a subclass of Pydantic BaseModel."
        )
    model: Union[Type[BaseModel], Type[pv1.BaseModel]] = test_res.model
    # If the passed function is a method of a class, test if the class is using a
    #  different Pydantic model version than the function, and eventually raise an error
    # compare_param_model_version(func, model_version)  # Not implemented yet

    # Create new parameters based on the fields of the params_type
    if model_version == PydanticModelVersion.V1:
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
        if func_type == FuncType.function:
            if len(args) == 1 and isinstance(args[0], model):
                # There is only one positional argument and it is an instance of the
                #  model
                return func(args[0])
            elif len(args) == 0 and isinstance(kwargs.get(param_name), model):
                # There are no positional arguments and the only keyword argument is an
                #  instance of the model
                return func(kwargs.get(param_name))
            else:
                if len(args) == 0:
                    # There are no positional arguments
                    # Create an instance of the model with the provided keyword
                    #  arguments
                    params_instance = model(**kwargs)
                else:
                    # There are positional and keyword arguments
                    # Pydantic does not allow positional arguments, so we need to map
                    #  the positional arguments to the field names of the model
                    # Get the name of potential arguments
                    if model_version == PydanticModelVersion.V1:
                        model__: Type[pv1.BaseModel] = model
                        pot_args = model__.__fields__.keys()
                    else:
                        model__: Type[BaseModel] = model
                        pot_args = model__.model_fields.keys()
                    # Create a dictionary with the positional arguments mapped to the
                    #  fields
                    new_args = dict(zip(pot_args, args))
                    # Merge the positional arguments with the keyword arguments
                    #  and create an instance of the model
                    params_instance = model(**{**new_args, **kwargs})
                # Call the original function with the model instance as the only
                #  argument
                return func(params_instance)
        else:
            # The function is a method of a class
            self_or_cls = args[0]
            if len(args) == 2 and isinstance(args[1], model):
                # There is only one positional argument and it is an instance of the
                #  model
                return func(self_or_cls, args[1])
            elif len(args) == 1 and isinstance(kwargs.get(param_name), model):
                # There are no positional arguments and the only keyword argument is an
                #  instance of the model
                return func(self_or_cls, kwargs.get(param_name))
            else:
                if len(args) == 1:
                    # There are no positional arguments
                    # Create an instance of the model with the provided keyword
                    #  arguments
                    params_instance = model(**kwargs)
                else:
                    # There are positional and keyword arguments
                    # Pydantic does not allow positional arguments, so we need to map
                    #  the positional arguments to the field names of the model
                    # Get the name of potential arguments
                    if model_version == PydanticModelVersion.V1:
                        model__: Type[pv1.BaseModel] = model
                        pot_args = model__.__fields__.keys()
                    else:
                        model__: Type[BaseModel] = model
                        pot_args = model__.model_fields.keys()
                    # Create a dictionary with the positional arguments mapped to the
                    #  fields
                    new_args = dict(zip(pot_args, args[1:]))
                    # Merge the positional arguments with the keyword arguments
                    #  and create an instance of the model
                    params_instance = model(**{**new_args, **kwargs})
                # Call the original function with the model instance as the only
                #  argument
                return func(self_or_cls, params_instance)

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

    def decorator(func: Callable):
        """Inner decorator to decorate the function

        Parameters
        ----------
        func
            The function to be decorated.
        """
        original_sig = inspect.signature(func)
        if len(original_sig.parameters) == 0:
            raise ValueError(
                "The passed function must have at least one parameter for "
                "this decorator to be applicable."
            )
        # Determine the Pydantic model version used in the function signature
        model_version = pydantic_model_version_used(func)
        # Create a Pydantic model from the function signature
        model = model_from_signature(
            func, caller_globals, model_name, model_version=model_version
        )
        if model_version == PydanticModelVersion.V1:
            dict_or_dump = "dict"
        else:
            dict_or_dump = "model_dump"
        # Create a new signature with the model as the only parameter
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
        # Test if the function is a method of a class
        func_type = function_or_method(func)
        # Test if the class is using a different Pydantic model version than the
        #  function, eventually raise an error
        # compare_param_model_version(func, model_version)  # Not implemented yet

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > 0 and func_type != FuncType.function:
                # print("The function is a method of class")
                self_or_cls = args[0]
                if len(args) == 2 and isinstance(args[1], model):
                    model_dump = getattr(args[1], dict_or_dump)
                    return func(self_or_cls, **model_dump())
                elif (
                    len(args) == 1
                    and len(kwargs) == 1
                    and isinstance(kwargs.get("params"), model)
                ):
                    model_dump = getattr(kwargs["params"], dict_or_dump)
                    return func(self_or_cls, **model_dump())
                else:
                    return func(*args, **kwargs)
            else:
                # print("The function is a standalone function")
                if len(args) == 1 and isinstance(args[0], model):
                    model_dump = getattr(args[0], dict_or_dump)
                    return func(**model_dump())
                elif (
                    len(args) == 0
                    and len(kwargs) == 1
                    and isinstance(kwargs.get("params"), model)
                ):
                    model_dump = getattr(kwargs["params"], dict_or_dump)
                    return func(**model_dump())
                else:
                    return func(*args, **kwargs)

        wrapper.__signature__ = new_sig
        return wrapper

    return decorator
