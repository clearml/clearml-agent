from typing import Callable, Dict, Any, Optional

_not_set = object()


def filter_keys(filter_, dct):  # type: (Callable[[Any], bool], Dict) -> Dict
    return {key: value for key, value in dct.items() if filter_(key)}


def merge_dicts(dict1, dict2, custom_merge_func=None):
    # type: (Any, Any, Optional[Callable[[str, Any, Any, Any], Any]]) -> Any
    """ Recursively merges dict2 into dict1 """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2
    for k in dict2:
        if k in dict1:
            res = _not_set
            if custom_merge_func:
                res = custom_merge_func(k, dict1[k], dict2[k], _not_set)
            dict1[k] = merge_dicts(dict1[k], dict2[k], custom_merge_func) if res is _not_set else res
        else:
            dict1[k] = dict2[k]
    return dict1
