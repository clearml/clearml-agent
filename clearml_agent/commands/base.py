from __future__ import unicode_literals, print_function

import copy
import re
import sys
from abc import abstractmethod
from functools import wraps
from operator import attrgetter
from traceback import print_exc
from time import time
from typing import Text

from clearml_agent.helper.console import ListFormatter

from .._vendor import six
from clearml_agent.backend_api import services

from clearml_agent.errors import APIError, CommandFailedError
from clearml_agent.helper.base import Singleton, return_list, print_parameters, dump_yaml, load_yaml, error, warning
from clearml_agent.interface.base import ObjectID
from clearml_agent.session import Session


class NameResolutionError(CommandFailedError):

    def __init__(self, message, suggestions=''):
        super(NameResolutionError, self).__init__(message)
        self.message = message
        self.suggestions = suggestions

    def __str__(self):
        return self.message + self.suggestions


def resolve_names(func):
    def safe_resolve(command, arg):
        try:
            result = command._resolve_name(arg.name, arg.service)
            return result, None
        except NameResolutionError:
            return arg.name, sys.exc_info()

    def _resolve_single(command, arg):
        if isinstance(arg, ObjectID):
            return command._resolve_name(arg.name, arg.service)
        elif isinstance(arg, (list, tuple)) and all(isinstance(x, ObjectID) for x in arg):
            result = [safe_resolve(command, x) for x in arg]
            if len(result) == 1:
                name, ex = result[0]
                if ex:
                    six.reraise(*ex)
                return [name]
            for _, ex in result:
                if ex:
                    command.warning(ex[1].message)
            return [name for (name, _) in result]
        return arg

    @wraps(func)
    def newfunc(self, *args, **kwargs):
        args = [_resolve_single(self, arg) for arg in args]
        kwargs = {key: _resolve_single(self, value) for key, value in kwargs.items()}
        return func(self, *args, **kwargs)
    return newfunc


class BaseCommandSection(object):
    """
    Base class for command sections which do not interact with the allegro API.
    Has basic utilities for user interaction.
    """
    warning = staticmethod(warning)
    error = staticmethod(error)

    @staticmethod
    def log(message, *args):
        print("clearml-agent: {}".format(message % args))

    @classmethod
    def exit(cls, message, code=1):  # type: (Text, int) -> ()
        cls.error(message)
        sys.exit(code)


@six.add_metaclass(Singleton)
class ServiceCommandSection(BaseCommandSection):
    """
    Base class for command sections which interact with the allegro API.
    Contains API functionality which is common across services.
    """

    _worker_name = None
    MAX_SUGGESTIONS = 10
    SERVICE_ID_NAME_LOOKUP_CACHE = {}  #  {id: str. dict(timestamp: float, name: str)}
    # todo: add cache size limit, currently only used for queus
    LOOKUP_CACHE_TIMEOUT_SEC = 180  # todo: make configurable from conf file,

    def __init__(self, *args, **kwargs):
        super(ServiceCommandSection, self).__init__()
        kwargs = self._verify_command_states(kwargs)
        self._session = self._get_session(*args, **kwargs)
        self._list_formatter = ListFormatter(self.service)

    @classmethod
    def _verify_command_states(cls, kwargs):
        """
        Conform and enforce command argument
        This is where you can automatically turn on/off switches based on different states.
        :param kwargs:
        :return: kwargs
        """
        return kwargs

    @staticmethod
    def _get_session(*args, **kwargs):
        return Session(*args, **kwargs)

    @property
    @abstractmethod
    def service(self):
        """ The name of the REST service used by this command """
        pass

    def get(self, endpoint, *args, service=None, session=None, **kwargs):
        session = session or self._session
        service = service or self.service
        return session.get(service=service, action=endpoint, *args, **kwargs)

    def post(self, endpoint, *args, service=None, session=None, **kwargs):
        session = session or self._session
        service = service or self.service
        return session.post(service=service, action=endpoint, *args, **kwargs)

    def get_with_act_as(self, endpoint, *args, **kwargs):
        return self._session.get_with_act_as(service=self.service, action=endpoint, *args, **kwargs)

    @property
    def name(self):
        return self.service.title()

    @property
    def name_single(self):
        return self.name.rstrip('s')

    @property
    def service_single(self):
        return self.service.rstrip('s')

    @resolve_names
    def __info(self, id=None, yaml=None, **kwargs):
        ids = return_list(id)
        if not ids:
            return

        yaml_dump = {}

        for i in ids:
            get_fields = {self.service_single: i}
            try:
                info = self.get('get_by_id', **get_fields)
                yaml_dump[i] = info[self.service_single]
            except APIError:
                self.error('Failed retrieving info for {} {}'.format(self.service_single, i))

        self.output_info(yaml_dump, yaml_path=yaml, **kwargs)
        return yaml_dump

    @resolve_names
    def _info(self, *args, **kwargs):
        self.__info(*args, **kwargs)

    @staticmethod
    def output_info(entries, quiet=False, yaml_path=None, **_):
        if not quiet and entries:
            print_parameters(entries, indent=4)

        if yaml_path:
            print('Storing entries to [{}]'.format(yaml_path))
            dump_yaml(entries, yaml_path)

    @staticmethod
    def _make_query(json, table, sort=None, projection_from_table=False, extra_fields=None):
        json = json.copy()
        if isinstance(table, six.string_types):
            table = table.split('#')

        if extra_fields:
            table.extend(extra_fields)

        if projection_from_table:
            json['only_fields'] = table

        if sort:
            # does nothing if 'order_by' is not in get_fields
            json['order_by'] = sort.split('#')[0]
        return json, table

    def _get_all(self, endpoint, json, retpoint=None):
        return self.get(endpoint, **json).get(retpoint or self.service, [])

    @resolve_names
    def _update(self,
                endpoint='update',
                send_diff=False,
                quiet=False,
                primary_key='id',
                override=None,
                model_desc=None,
                yaml=None,
                **kwargs):

        if not yaml and primary_key not in kwargs:
            raise ValueError('Update must supply either yaml file or %s-id' % self.service_single)

        data_entries = {}
        original_data_entries = {}
        if yaml:
            data_entries = load_yaml(yaml)

        if send_diff or (not yaml and primary_key in kwargs):
            i = kwargs.get(primary_key) or next(iter(data_entries))
            original_info = self.__info(id=i, quiet=True)[i]

            if send_diff:
                original_data_entries[i] = copy.deepcopy(original_info)
            if yaml and i not in data_entries:
                if len(data_entries) > 1:
                    raise ValueError(
                        'Error: yaml file [%s] contains more than one task id' % yaml)
                first_key = next(iter(data_entries))
                if first_key != i:
                    if kwargs.get('force'):
                        if not quiet:
                            print('Warning: overriding yaml task id [%s] with id=%s' % (first_key, i))
                    else:
                        raise ValueError(
                            'Error: yaml task id [%s] != id [%s], use --force to override' % (first_key, i))
                    data_entries = {i: data_entries[first_key]}
                    data_entries[i][primary_key] = i
            elif not yaml:
                data_entries[i] = kwargs

        if model_desc:
            first_key = next(iter(data_entries))
            with open(model_desc) as f:
                proto_data = f.read()
                info = data_entries[first_key]
                info['execution']['model_desc']['prototxt'] = proto_data

        if override:
            first_key = next(iter(data_entries))
            info = data_entries[first_key]
            for p in override:
                key, val = p.split('=') if isinstance(p, six.string_types) else p
                info_key = info
                keys = key.split('.')
                for k in keys[:-1]:
                    if not info_key.get(k):
                        info_key[k] = dict()
                    info_key = info_key[k]
                info_key[keys[-1]] = val

            # always make sure tags is a list of strings
            # split string to tokens ':'
            # examples tags='auto_generated:draft'
            if info.get('tags') is not None:
                # remove empty strings from list
                info['tags'] = [t for t in info['tags'].split(':') if t]

        # send only change set
        if send_diff:
            # only send the values that changed
            for i, info in data_entries.items():
                org_info = original_data_entries[i]
                out_info = {}
                recursive_diff(org_info, info, out_info)
                data_entries[i] = out_info

        for i, info in data_entries.items():
            if not info or len(info) == 0 or list(info) == [primary_key]:
                if not quiet:
                    print('Skipping: nothing to update for %s id [%s]' % (self.service_single, i))
                continue
            if not quiet:
                print('Updating %s id [%s]' % (self.service_single, i))
            info[self.service_single] = i
            result = self.get(endpoint, **info)
            if not result['updated']:
                raise ValueError('Failed updating %s id [%s]' % (self.service_single, i))
            if not quiet:
                print('%s [%s] updated fields: %s' % (self.name_single, i, result.get('fields', '')))

    @resolve_names
    def remove(self, ids, **kwargs):
        return self._apply_command(
            request_cls=getattr(services, self.service).DeleteRequest,
            object_ids=ids,
            response_validation_field='deleted',
            **kwargs
        )

    def _apply_command(self, request_cls, object_ids, response_validation_field=None, **kwargs):
        object_ids = return_list(object_ids)

        def call_one(object_id):
            error_message = '[{object_id}]: failed'.format(**locals())
            try:
                response = self._session.send_api(request_cls(object_id, **kwargs))
            except APIError as e:
                if not self._session.debug_mode:
                    self.error('{}: {}'.format(error_message, e))
                else:
                    traceback = e.format_traceback()
                    if traceback:
                        print(traceback)
                        print('Own traceback:')
                    print_exc()
                return False

            if not response_validation_field or getattr(response, response_validation_field) == 1:
                return True
            else:
                self.error(error_message)
                return False

        succeeded = [call_one(object_id) for object_id in object_ids].count(True)
        message = '{}/{} succeeded'.format(succeeded, len(object_ids))
        (self.log if succeeded == len(object_ids) else self.exit)(message)

    def get_service(self, service_class):
        return service_class(config=self._session.config)

    def _resolve_name(self, name, service=None, search_hidden=False):
        """
        Resolve an object name to an object ID.
        Operation:
        - If the argument "looks like" an ID, return it.
        - Else, get all object with names containing the argument
            - if an object with the argument as its name exists, return the object's ID
            - Else, print a list of suggestions and exit
        :param str name: ID (returned unmodified) or Name to resolve
        :param str service: Service to resolve from (type of object). Defaults to service represented by the class
        :return: ID of object
        :rtype: str
        """
        service = service or self.service
        if re.match(r'^[-a-f0-9]{30,}$', name):
            return name

        try:
            request_cls = getattr(services, service).GetAllRequest
        except AttributeError:
            raise NameResolutionError('Name resolution unavailable for {}'.format(service))

        req_dict = {"name": re.escape(name), "only_fields": ['name', 'id']}
        if search_hidden:
            req_dict["_allow_extra_fields_"] = True
            req_dict["search_hidden"] = True
        request = request_cls.from_dict(req_dict)
        # from_dict will ignore unrecognised keyword arguments - not all GetAll's have only_fields
        response = getattr(self._session.send_api(request), service)
        matches = [db_object for db_object in response if name.lower() == db_object.name.lower()]

        def truncated_bullet_list(format_string, elements, callback, **kwargs):
            if len(elements) > self.MAX_SUGGESTIONS:
                kwargs.update(
                    dict(details=' (showing {}/{})'.format(self.MAX_SUGGESTIONS, len(elements)), suffix='\n...'))
            else:
                kwargs.update(dict(details='', suffix=''))
            bullet_list = '\n'.join('* {}'.format(callback(item)) for item in elements[:self.MAX_SUGGESTIONS])
            return format_string.format(bullet_list, **kwargs)

        if len(matches) == 1:
            return matches.pop().id
        elif len(matches) > 1:
            message = truncated_bullet_list(
                'Found multiple {service} with name "{name}"{details}:\n{}{suffix}',
                matches,
                callback=attrgetter('id'),
                **locals())
            self.exit(message)

        message = 'Could not find {} with name/id "{}"'.format(service.rstrip('s'), name)

        if not response:
            raise NameResolutionError(message)

        suggestions = truncated_bullet_list(
            '. Did you mean this?{details}\n{}{suffix}',
            sorted(response, key=attrgetter('name')),
            lambda db_object: '({}) {}'.format(db_object.id, db_object.name)
        )
        raise NameResolutionError(message, suggestions)

    def _resolve_id(self, id, service=None, search_hidden=False):
        """
        Resolve an object ID to an object name. (ID is UUID in the system, regardless of service)
        Operation:
        - If the argument "looks like" an NAME, return it.
        - Else, get all object with names containing the argument
            - if an object with the argument as its name exists, return the object's NAME
            - Else, print a list of suggestions and exit
        :param str id: NAME (returned unmodified) or ID to resolve
        :param str service: Service to resolve from (type of object). Defaults to service represented by the class
        :return: name of object
        :rtype: str
        """
        service = service or self.service
        # check if this is a name not an ID
        if not re.match(r'^[-a-f0-9]{30,}$', id):
            return id

        cache_entry = self.SERVICE_ID_NAME_LOOKUP_CACHE.get(id)
        if cache_entry:
            if time() - cache_entry['timestamp'] < self.LOOKUP_CACHE_TIMEOUT_SEC:
                return cache_entry['name']

        try:
            request_cls = getattr(services, service).GetAllRequest
        except AttributeError:
            raise NameResolutionError('Name resolution unavailable for {}'.format(service))

        req_dict = {"id": [re.escape(id)], "only_fields": ['name', 'id']}
        if search_hidden:
            req_dict["_allow_extra_fields_"] = True
            req_dict["search_hidden"] = True
        request = request_cls.from_dict(req_dict)
        # from_dict will ignore unrecognised keyword arguments - not all GetAll's have only_fields
        response = getattr(self._session.send_api(request), service)
        matches = [db_object for db_object in response if id.lower() == db_object.id.lower()]

        if len(matches) == 1:
            cache_entry = dict(timestamp=time(), name=matches.pop().name)
            self.SERVICE_ID_NAME_LOOKUP_CACHE[id] = cache_entry
            return cache_entry['name']
        elif len(matches) > 1:
            raise NameResolutionError('Found multiple {} with Id "{}" (#{})'.format(service, id, len(matches)))

        message = 'Could not find {} with name/id "{}"'.format(service.rstrip('s'), id)
        raise NameResolutionError(message)


def recursive_diff(org, upd, out):
    if isinstance(upd, dict) and isinstance(org, dict):
        diff_keys = [
            k for k in upd
            if k not in org or upd[k] != org[k]
        ]
        if diff_keys:
            has_nested_dict = False
            for k in diff_keys:
                if isinstance(upd[k], dict):
                    out[k] = {}
                    has_nested_dict = True
                    k_has_nested = recursive_diff(
                        org.get(k, {}), upd[k], out[k])
                    if not k_has_nested:
                        out[k] = upd[k]
                elif upd[k] is not None:
                    out[k] = upd[k]
            return has_nested_dict
    elif isinstance(upd, list) and isinstance(org, list):
        diff_list = [k for k in upd if k not in org]
        out.extend(diff_list)
    return False
