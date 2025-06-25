from ._vendor import six

from clearml_agent.external.pyhocon import ConfigTree
from clearml_agent.helper.base import Singleton


@six.add_metaclass(Singleton)
class Config(object):

    def __init__(self, tree=None):
        self.__dict__['_tree'] = tree or ConfigTree()

    def __getitem__(self, item):
        return self._tree[item]

    def __setitem__(self, key, value):
        return self._tree.__setitem__(key, value)

    def new(self, name):
        return self._tree.setdefault(name, ConfigTree())

    __getattr__ = __getitem__
    __setattr__ = __setitem__


def get_config(name=None):
    config = Config()
    if name:
        return getattr(config, name)
    return config


def make_config(name):
    return get_config().new(name)
