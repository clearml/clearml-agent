import sys
import time

from ...backend_api.utils import get_response_cls

from .response import ResponseMeta, Response
from .errors import ResultNotReadyError, TimeoutExpiredError


class CallResult(object):
    @property
    def meta(self):
        """
        Returns the meta - meta.

        Args:
            self: (todo): write your description
        """
        return self.__meta

    @property
    def response(self):
        """
        : return the response.

        Args:
            self: (todo): write your description
        """
        return self.__response

    @property
    def response_data(self):
        """
        Return the response data.

        Args:
            self: (todo): write your description
        """
        return self.__response_data

    @property
    def async_accepted(self):
        """
        Return the result of the result.

        Args:
            self: (todo): write your description
        """
        return self.meta.result_code == 202

    @property
    def request_cls(self):
        """
        : return request class.

        Args:
            self: (todo): write your description
        """
        return self.__request_cls

    def __init__(self, meta, response=None, response_data=None, request_cls=None, session=None):
        """
        Initialize the response.

        Args:
            self: (todo): write your description
            meta: (float): write your description
            response: (list): write your description
            response_data: (todo): write your description
            request_cls: (todo): write your description
            session: (todo): write your description
        """
        assert isinstance(meta, ResponseMeta)
        if response and not isinstance(response, Response):
            raise ValueError('response should be an instance of %s' % Response.__name__)
        elif response_data and not isinstance(response_data, dict):
            raise TypeError('data should be an instance of {}'.format(dict.__name__))

        self.__meta = meta
        self.__response = response
        self.__request_cls = request_cls
        self.__session = session
        self.__async_result = None

        if response_data is not None:
            self.__response_data = response_data
        elif response is not None:
            try:
                self.__response_data = response.to_dict()
            except AttributeError:
                raise TypeError('response should be an instance of {}'.format(Response.__name__))
        else:
            self.__response_data = None

    @classmethod
    def from_result(cls, res, request_cls=None, logger=None, service=None, action=None, session=None):
        """ From requests result """
        response_cls = get_response_cls(request_cls)
        try:
            data = res.json()
        except ValueError:
            service = service or (request_cls._service if request_cls else 'unknown')
            action = action or (request_cls._action if request_cls else 'unknown')
            return cls(request_cls=request_cls, meta=ResponseMeta.from_raw_data(
                status_code=res.status_code, text=res.text, endpoint='%(service)s.%(action)s' % locals()))
        if 'meta' not in data:
            raise ValueError('Missing meta section in response payload')
        try:
            meta = ResponseMeta(**data['meta'])
            # TODO: validate meta?
            # meta.validate()
        except Exception as ex:
            raise ValueError('Failed parsing meta section in response payload (data=%s, error=%s)' % (data, ex))

        response = None
        response_data = None
        try:
            response_data = data.get('data', {})
            if response_cls:
                response = response_cls(**response_data)
                # TODO: validate response?
                # response.validate()
        except Exception as e:
            if logger:
                logger.warning('Failed parsing response: %s' % str(e))
        return cls(meta=meta, response=response, response_data=response_data, request_cls=request_cls, session=session)

    def ok(self):
        """
        Determine if the result.

        Args:
            self: (todo): write your description
        """
        return self.meta.result_code == 200

    def ready(self):
        """
        Determine request is ready.

        Args:
            self: (todo): write your description
        """
        if not self.async_accepted:
            return True
        session = self.__session
        res = session.send_request(service='async', action='result', json=dict(id=self.meta.id), async_enable=False)
        if res.status_code != session._async_status_code:
            self.__async_result = CallResult.from_result(res=res, request_cls=self.request_cls, logger=session._logger)
            return True

    def result(self):
        """
        Return the result of the result.

        Args:
            self: (todo): write your description
        """
        if not self.async_accepted:
            return self
        if self.__async_result is None:
            raise ResultNotReadyError(self._format_msg('Timeout expired'), call_id=self.meta.id)
        return self.__async_result

    def wait(self, timeout=None, poll_interval=5, verbose=False):
        """
        Waits for a command to complete.

        Args:
            self: (todo): write your description
            timeout: (float): write your description
            poll_interval: (int): write your description
            verbose: (bool): write your description
        """
        if not self.async_accepted:
            return self
        session = self.__session
        poll_interval = max(1, poll_interval)
        remaining = max(0, timeout) if timeout else sys.maxsize
        while remaining > 0:
            if not self.ready():
                # Still pending, log and continue
                if verbose and session._logger:
                    progress = ('waiting forever'
                                if timeout is False
                                else '%.1f/%.1f seconds remaining' % (remaining, float(timeout or 0)))
                    session._logger.info('Waiting for asynchronous call %s (%s)'
                                         % (self.request_cls.__name__, progress))
                time.sleep(poll_interval)
                remaining -= poll_interval
                continue
            # We've got something (good or bad, we don't know), create a call result and return
            return self.result()

        # Timeout expired, return the asynchronous call's result (we've got nothing better to report)
        raise TimeoutExpiredError(self._format_msg('Timeout expired'), call_id=self.meta.id)

    def _format_msg(self, msg):
        """
        Formats the message. message.

        Args:
            self: (todo): write your description
            msg: (str): write your description
        """
        return msg + ' for call %s (%s)' % (self.request_cls.__name__, self.meta.id)
