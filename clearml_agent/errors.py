from typing import Union, Optional, Text

import requests
from ._vendor import six
from .backend_api import CallResult
from .backend_api.session.client import APIError as ClientAPIError
from .backend_api.session.response import ResponseMeta

INTERNAL_SERVER_ERROR = 500


# TODO: hack: should NOT inherit from ValueError
class APIError(ClientAPIError, ValueError):
    """
    Class for representing an API error.

    self.data - ``dict`` of all returned JSON data
    self.code - HTTP response code
    self.subcode - server response subcode
    self.codes - (self.code, self.subcode) tuple
    self.message - result message sent from server
    """

    def __init__(self, response, extra_info=None):
        # type: (Union[requests.Response, CallResult], Optional[Text]) -> None
        """
        Create a new APIError from a server response
        """
        if not isinstance(response, CallResult):
            try:
                data = response.json()
            except ValueError:
                data = {}
            meta = data.get('meta')
            if meta:
                response_meta = ResponseMeta(is_valid=False, **meta)
            else:
                response_meta = ResponseMeta.from_raw_data(response.status_code, response.text)
            response = CallResult(
                meta=response_meta,
                response=response,
                response_data=data,
            )
        super(APIError, self).__init__(response, extra_info=extra_info)

    def format_traceback(self):
        if self.code != INTERNAL_SERVER_ERROR:
            return ''
        traceback = self.get_traceback()
        if traceback:
            return 'Server traceback:\n{}'.format(traceback)
        else:
            return 'Could not print server traceback'


class CommandFailedError(Exception):

    def __init__(self, message=None, *args, **kwargs):
        super(CommandFailedError, self).__init__(message, *args, **kwargs)
        self.message = message


class UsageError(CommandFailedError):
    """
    Used for usage errors that are checked post-argparsing
    """
    pass


class ConfigFileNotFound(CommandFailedError):
    pass


class Sigterm(BaseException):
    pass


@six.python_2_unicode_compatible
class MissingPackageError(CommandFailedError):
    def __init__(self, name):
        super(MissingPackageError, self).__init__(name)
        self.name = name

    def __str__(self):
        return '{self.__class__.__name__}: ' \
               '"{self.name}" package is required. Please run "pip install {self.name}"'.format(self=self)


class CustomBuildScriptFailed(CommandFailedError):
    def __init__(self, errno, *args, **kwargs):
        super(CustomBuildScriptFailed, self).__init__(*args, **kwargs)
        self.errno = errno


class SkippedCustomBuildScript(CommandFailedError):
    pass
