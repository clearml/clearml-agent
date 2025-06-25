import base64
from typing import Union, Optional, Any, TypeVar, Callable, Tuple

from ..._vendor import six

try:
    from typing import Text
except ImportError:
    # windows conda-less hack
    Text = Any


ConverterType = TypeVar("ConverterType", bound=Callable[[Any], Any])


def base64_to_text(value):
    # type: (Any) -> Text
    return base64.b64decode(value).decode("utf-8")


def text_to_int(value, default=0):
    # type: (Any, int) -> int
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def text_to_bool(value):
    # type: (Text) -> bool
    return bool(strtobool(value))


def safe_text_to_bool(value):
    # type: (Text) -> bool
    try:
        return text_to_bool(value)
    except ValueError:
        return bool(value)


def any_to_bool(value):
    # type: (Optional[Union[int, float, Text]]) -> bool
    if isinstance(value, six.text_type):
        return text_to_bool(value)
    return bool(value)


# noinspection PyIncorrectDocstring
def or_(*converters, **kwargs):
    # type: (ConverterType, Tuple[Exception, ...]) -> ConverterType
    """
    Wrapper that implements an "optional converter" pattern. Allows specifying a converter
    for which a set of exceptions is ignored (and the original value is returned)
    :param converters: A converter callable
    :param exceptions: A tuple of exception types to ignore
    """
    # noinspection PyUnresolvedReferences
    exceptions = kwargs.get("exceptions", (ValueError, TypeError))

    def wrapper(value):
        for converter in converters:
            try:
                return converter(value)
            except exceptions:
                pass
        return value

    return wrapper


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))
