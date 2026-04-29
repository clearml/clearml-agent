# -*- coding: utf-8 -*-

#
# omdict - Ordered Multivalue Dictionary.
#
# Ansgar Grunseid
# grunseid.com
# grunseid@gmail.com
#
# License: Build Amazing Things (Unlicense)
#

from os.path import dirname, join as pjoin

from .orderedmultidict import *  # noqa

# removed this horrible code and explicitly import metadata
# # Import all variables in __version__.py without explicit imports.
# meta = {}
# with open(pjoin(dirname(__file__), '__version__.py')) as f:
#     exec(f.read(), meta)
# globals().update(dict((k, v) for k, v in meta.items() if k not in globals()))
from .__version__ import __title__, __version__, __license__, __author__ , __contact__, __description__, __url__
