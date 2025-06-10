from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
)

# Because we need to type our own stuff, we have to make everything from
# attr explicitly public too.
from ..._vendor.attr import __author__ as __author__
from ..._vendor.attr import __copyright__ as __copyright__
from ..._vendor.attr import __description__ as __description__
from ..._vendor.attr import __email__ as __email__
from ..._vendor.attr import __license__ as __license__
from ..._vendor.attr import __title__ as __title__
from ..._vendor.attr import __url__ as __url__
from ..._vendor.attr import __version__ as __version__
from ..._vendor.attr import __version_info__ as __version_info__
from ..._vendor.attr import _FilterType
from ..._vendor.attr import assoc as assoc
from ..._vendor.attr import Attribute as Attribute
from ..._vendor.attr import AttrsInstance as AttrsInstance
from ..._vendor.attr import cmp_using as cmp_using
from ..._vendor.attr import converters as converters
from ..._vendor.attr import define as define
from ..._vendor.attr import evolve as evolve
from ..._vendor.attr import exceptions as exceptions
from ..._vendor.attr import Factory as Factory
from ..._vendor.attr import field as field
from ..._vendor.attr import fields as fields
from ..._vendor.attr import fields_dict as fields_dict
from ..._vendor.attr import filters as filters
from ..._vendor.attr import frozen as frozen
from ..._vendor.attr import has as has
from ..._vendor.attr import make_class as make_class
from ..._vendor.attr import mutable as mutable
from ..._vendor.attr import NOTHING as NOTHING
from ..._vendor.attr import resolve_types as resolve_types
from ..._vendor.attr import setters as setters
from ..._vendor.attr import validate as validate
from ..._vendor.attr import validators as validators

# TODO: see definition of attr.asdict/astuple
def asdict(
    inst: Any,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    dict_factory: Type[Mapping[Any, Any]] = ...,
    retain_collection_types: bool = ...,
    value_serializer: Optional[
        Callable[[type, Attribute[Any], Any], Any]
    ] = ...,
    tuple_keys: bool = ...,
) -> Dict[str, Any]: ...

# TODO: add support for returning NamedTuple from the mypy plugin
def astuple(
    inst: Any,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    tuple_factory: Type[Sequence[Any]] = ...,
    retain_collection_types: bool = ...,
) -> Tuple[Any, ...]: ...
