"""MD5 helper compatible with FIPS-enabled Python builds.

`usedforsecurity=False` was added in Python 3.9; clearml-agent declares
support for 3.6+, so we feature-detect once and dispatch.
"""
import hashlib
import sys

_HAS_USEDFORSECURITY = sys.version_info >= (3, 9)


def md5(data=b"", usedforsecurity=False):
    # type: (bytes, bool) -> "hashlib._Hash"
    """Return an md5 hasher, defaulting to non-security use where supported.

    The digest is identical with or without the flag; it only affects
    FIPS policy enforcement at the hashlib layer.
    """
    if _HAS_USEDFORSECURITY:
        return hashlib.md5(data, usedforsecurity=usedforsecurity)
    return hashlib.md5(data)
