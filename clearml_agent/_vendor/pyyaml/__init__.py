from ..ruamel.yaml import YAML
from ..ruamel.yaml.error import YAMLError as RuamelYAMLError


# Initialize a global ruamel.yaml YAML instance
yaml_instance = YAML(typ='safe')  # Ensuring we use the safe loader


# Define a custom YAMLError to maintain compatibility with pyyaml
class YAMLError(Exception):
    pass


# compatibility
class SafeDumper(object):
    pass


def load(stream, Loader=None):
    """
    Load YAML content from a stream using ruamel.yaml

    :param stream: Input stream (file-like object or string)
    :param Loader: Placeholder to maintain compatibility, not used
    :return: Parsed content from the YAML file
    """
    try:
        data = yaml_instance.load(stream)
        return data
    except RuamelYAMLError as e:
        raise YAMLError(str(e))


def dump(data, stream=None, Dumper=None, **kwargs):
    """
    Dump Python objects to a YAML-formatted stream using ruamel.yaml

    :param data: Python object to be serialized
    :param stream: Output stream (file-like object or None). Dumps to a string if None.
    :param Dumper: Placeholder to maintain compatibility, not used
    :return: If stream is None, returns the produced YAML string
    """
    try:
        if stream is None:
            from io import StringIO
            stream = StringIO()
            yaml_instance.dump(data, stream)
            return stream.getvalue()
        yaml_instance.dump(data, stream)
    except RuamelYAMLError as e:
        raise YAMLError(str(e))


# Alias for safe_dump to maintain compatibility
safe_dump = dump


def safe_load(stream):
    """
    Load YAML content from a stream safely using ruamel.yaml

    :param stream: Input stream (file-like object or string)
    :return: Parsed content from the YAML file
    """
    try:
        data = yaml_instance.load(stream)
        return data
    except RuamelYAMLError as e:
        raise YAMLError(str(e))

