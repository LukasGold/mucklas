import inspect
from functools import partialmethod, wraps
from typing import Any, Callable, Dict, Type, TypeVar, Union, get_args

import pydantic.v1 as pv1
from pydantic import BaseModel, ConfigDict, create_model
from typing_extensions import deprecated

T = TypeVar("T")


def partialclass(cls, *args, **kwargs):
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


def pydantic_model(cls: Type[T]) -> Type[BaseModel]:
    """Decorator to be used on a class definition to create a Pydantic (v2) model
    from a dataclass."""
    return create_model(cls.__name__, __base__=(BaseModel, cls), __config__=None)


def pydantic_v1_model(cls: Type[T]) -> Type[pv1.BaseModel]:
    """Decorator to be used on a class definition to create a Pydantic (v1) model
    from a dataclass."""
    return pv1.create_model(cls.__name__, __base__=(BaseModel, cls), __config__=None)


def model_from_signature(
    func: Callable,
    model_name: str = None,
    model_description: str = None,
    strict: bool = False,
) -> Type[BaseModel]:
    """Function to create a Pydantic model from a function signature, using type
    hints to annotate the model fields."""
    # Extract the function signature
    signature = inspect.signature(func)
    model_fields: Dict[str, Any] = {
        name: (
            param.annotation,
            param.default if param.default is not inspect.Parameter.empty else ...,
        )
        for name, param in signature.parameters.items()
    }
    # create mode, set validation to strict
    if strict is not False:
        model_fields["model_config"] = ConfigDict(
            strict=strict, arbitrary_types_allowed=True
        )
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
    model_fields["__doc__"] = model_description
    # Create a Pydantic model dynamically
    return create_model(model_name, **model_fields)


def replace_param(func: Callable) -> Callable:
    """Decorator that will modify the function signature to accept keyword arguments
    corresponding to the fields of a Pydantic model. The Pydantic model is expected
    to be the only parameter of the function."""
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
    v1 = False
    model: Union[Type[BaseModel], Type[pv1.BaseModel]] = None
    # Ensure params_type is a subclass of BaseModel
    if (
        getattr(param_type, "__origin__", None) is not None
        and getattr(param_type, "__origin__", None) == Union
    ):
        # Test if it is a Union with a BaseModel
        for arg in get_args(param_type):
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                model: Type[BaseModel] = arg
                break
            elif isinstance(arg, type) and issubclass(arg, pv1.BaseModel):
                model: Type[pv1.BaseModel] = arg
                v1 = True
                break
    elif issubclass(param_type, BaseModel):
        # Test if param_type is a subclass of BaseModel
        model: Type[BaseModel] = param_type
    elif issubclass(param_type, pv1.BaseModel):
        model: Type[pv1.BaseModel] = param_type
        v1 = True
    else:
        raise TypeError(
            f"The type hint of the single parameter of the function '{func.__name__}' "
            "must be a subclass of Pydantic BaseModel. If the type hint is a Union, "
            "one argument of Union must be a subclass of Pydantic BaseModel."
        )
    # Create new parameters based on the fields of the params_type
    if v1:
        model: Type[pv1.BaseModel]
        new_params = [
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=field.default
            )
            for name, field in model.__fields__.items()
        ]
    else:
        model: Type[BaseModel]
        new_params = [
            inspect.Parameter(
                name, inspect.Parameter.KEYWORD_ONLY, default=field.default
            )
            for name, field in model.model_fields.items()
        ]

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Unfortunately, it is not possible to programmatically overload the decorated
        function with a signature that contains kwargs and at the same time keep the
        signature with the model instance as the only argument."""
        if len(args) == 1 and isinstance(args[0], model):
            # Call the original function with the model instance as the only argument
            return func(args[0])
        elif len(args) == 0 and isinstance(kwargs.get(param_name), model):
            # Call the original function with the model instance as the only argument
            return func(kwargs.get(param_name))
        else:
            if len(args) == 0:
                # Create an instance of the model with the provided keyword arguments
                params_instance = model(**kwargs)
            else:
                # Pydantic does not allow positional arguments, so we need to map the
                # positional arguments to the field names of the model
                if v1:
                    # model: Type[pv1.BaseModel]
                    pot_args = model.__fields__.keys()
                else:
                    # model: Type[BaseModel]
                    pot_args = model.model_fields.keys()
                new_args = dict(zip(pot_args, args))

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
