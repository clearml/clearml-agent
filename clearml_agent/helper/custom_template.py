import logging
import re
from collections.abc import Mapping
from copy import deepcopy
from functools import partial
from string import Template
from typing import Optional, Callable, Union, List, Tuple, Dict

from clearml_agent.backend_api.services import queues as queues_api
from clearml_agent.backend_api.session import Request
from clearml_agent.external.pyhocon import ConfigTree
from clearml_agent.helper.task_logs import send_logs_to_task
from .._vendor import pyyaml as yaml

log = logging.getLogger(__name__)


class LazyDict(Mapping):
    NotSet = object()

    def __init__(self, name: str, load_cb: Callable[[], Union[ConfigTree, dict]], error_cb: Callable[[str], None]):
        self.name = name
        self.load_cb = load_cb
        self.error_cb = error_cb
        self._dict = self.NotSet

    def load(self) -> Union[ConfigTree, dict]:
        if self._dict is self.NotSet:
            try:
                self._dict = self.load_cb()
            except Exception as e:
                self.error_cb(f"Failed lazy loading {self.name}: {e}")
                self._dict = {}
        return self._dict

    def __getitem__(self, key):
        return self.load().__getitem__(key)

    def __iter__(self):
        return iter(self.load())

    def __len__(self):
        return len(self.load())


class CustomTemplate(Template):
    """
    Parse ${CLEARML_<something>:default} values based on Task object and replace with real-time value
    Example: "-e project_id=${CLEARML_TASK.project}" will be replaced with the
             Task actual project ID from the Task object "-e project_id=<task.project>"
             "-e queue_name=${CLEARML_QUEUE_NAME}"
             "-e user_key=${TASK.hyperparams.properties.user_key.value:default_value}"

    Supported:

        ${QUEUE_NAME}
        ${WORKER_ID}
        ${QUEUE_ID}

        Complex variables are also supported:

        ${TASK.id}
        ${TASK.name}
        ${TASK.project.id}
        ${TASK.project.name}
        ${TASK.hyperparams.properties.user_key.value}
        ${PROVIDERS_INFO...}
        ${CONFIG...}
        ${USER_VAULTS...}
        ${USER...}
    """

    delimiter = '$'
    idpattern = r'(?a:[_a-z][_a-z0-9|.|:]*)'
    prefix = "CLEARML_"
    filter_sep = "|"
    filter_re = re.compile(r"^(?P<op>[^(]+)(\((?P<args>[^()]*)\))?")
    queue_id_to_name_map = {}

    remove_newlines_op = "remove_newlines"
    keep_newlines_op = "raw"
    function_ops = {
        "strip": lambda s, args: s.strip(*args),
        "capitalize": lambda s, _: s.capitalize(),
        "lower": lambda s, _: s.lower(),
        remove_newlines_op: lambda s, _: s.replace("\n", ""),
    }
    default_ops = [remove_newlines_op, ]

    @classmethod
    def get_queue_name(cls, task_session, queue_id):
        if queue_id in cls.queue_id_to_name_map:
            return cls.queue_id_to_name_map[queue_id]

        # noinspection PyBroadException
        try:
            response = task_session.send_api(queues_api.GetByIdRequest(queue=queue_id))
            cls.queue_id_to_name_map[queue_id] = response.queue.name
        except Exception:
            # if something went wrong start over from the highest priority queue
            return None
        return cls.queue_id_to_name_map.get(queue_id)

    def __init__(self, template, support_ops=True):
        super().__init__(template)
        self._support_ops = support_ops

    def default_custom_substitute(
        self,
        queue_id: str,
        queue_name: str,
        worker_id: str,
        task_info: Mapping,
        provider_info: Mapping = None,
        config: Mapping = None,
        user_vaults: Mapping = None,
        user_info: Mapping = None
    ):
        return self.custom_substitute(
            partial(
                CustomTemplate.default_resolve_template,
                queue_id=queue_id,
                queue_name=queue_name,
                worker_id=worker_id,
                task_info=task_info,
                provider_info=provider_info,
                config=config,
                user_vaults=user_vaults,
                user_info=user_info,
            )
        )

    def _apply_ops(self, ops, value):
        skip = set()
        for op, args in (x if isinstance(x, tuple) else (x, None) for x in ops):
            if op in skip:
                continue
            if op == self.keep_newlines_op:
                skip.add(self.remove_newlines_op)
                continue
            func = self.function_ops.get(op)
            if not func:
                continue
            try:
                args_ = args.split(",") if args else None
                value = func(value, args_)
            except Exception as e:
                log.error(f"Failed applying op {op} with args {args}: {e}")
        return value

    def custom_substitute(self, mapping_func, disable_ops_processing=False):
        # Helper function for .sub()
        def convert(mo):
            named = mo.group('named') or mo.group('braced')
            if not named or not str(named).startswith(self.prefix):
                return mo.group()
            named = named[len(self.prefix):]

            if named is not None:
                parsed_ops: List[Union[str, Tuple[str, str]]] = self.default_ops[:]
                if self._support_ops and self.filter_sep in named:
                    named, *ops = named.split(self.filter_sep)[1:]
                    parsed_ops.extend(
                        (match.group("op"), match.group("args") or None)
                        for match in (
                            self.filter_re.match(x) for x in ops if self.filter_re.match(x)
                        )
                    )

                default_value = None
                try:
                    if ":" in named:
                        named, default_value = named.split(":", 1)
                    result = str(mapping_func(named, default_value))
                    if not disable_ops_processing and parsed_ops:
                        result = self._apply_ops(parsed_ops, result)
                    return result
                except KeyError:
                    return mo.group()
            if mo.group('escaped') is not None:
                return self.delimiter
            if mo.group('invalid') is not None:
                return mo.group()
            raise ValueError('Unrecognized named group in pattern', self.pattern)

        return self.pattern.sub(convert, self.template)

    def substitute(self, *args, **kwds):
        raise ValueError("Unsupported")

    def safe_substitute(self, *args, **kwds):
        raise ValueError("Unsupported")

    @classmethod
    def default_resolve_template(
        cls,
        key: str,
        default: Optional[str],
        queue_id: str,
        queue_name: str,
        worker_id: str,
        task_info: Mapping,
        provider_info: Mapping = None,
        config: Mapping = None,
        user_vaults: Mapping = None,
        user_info: Mapping = None,
    ):
        """
        Notice CLEARML_ prefix omitted! (i.e. ${QUEUE_ID} is ${QUEUE_ID})

        Supported:

        ${QUEUE_NAME}
        ${WORKER_ID}
        ${QUEUE_ID}

        Complex variables are also supported:

        ${TASK.id}
        ${TASK.name}
        ${TASK.project.id}
        ${TASK.project.name}
        ${TASK.hyperparams.properties.user_key.value}
        ${PROVIDERS_INFO...}
        ${CONFIG...}
        ${USER_VAULTS...}
        ${USER...}

        :param user_info: nested dict with user information
        :param user_vaults: nested dict with user vaults configuration
        :param config: nested dict with complete configuration (includes vaults)
        :param provider_info: nested dict with provider information collected during user login (configurable in server)
        :param worker_id: agent worker ID (str)
        :param task_info: nested dict with task information
        :param queue_id: queue_id (str)
        :param queue_name: queue_name (str)
        :param key: key to be replaced
        :param default: default value, None will raise exception
        :return: string value
        """
        single_field_mapping = {
            "QUEUE_ID": queue_id,
            "QUEUE_NAME": queue_name,
            "WORKER_ID": worker_id,
        }
        nested_field_mapping = {
            "TASK": task_info or {},
            "PROVIDERS_INFO": provider_info or {},
            "CONFIG": config or {},
            "USER_VAULTS": user_vaults or {},
            "USER": user_info or {},
        }
        try:
            prefix, *path = key.split(".")
            if prefix in single_field_mapping:
                if path:
                    raise ValueError(f"Field {prefix} does not support any path")
                return single_field_mapping[prefix] or default
            elif prefix in nested_field_mapping:
                cur = nested_field_mapping[prefix] or {}
                for part in path:
                    cur = cur.get(part)
                    if cur is None:
                        break
                if isinstance(cur, str):
                    return cur
                elif isinstance(cur, bool):
                    return "true" if cur else "false"
                elif isinstance(cur, (int, float)):
                    return str(cur)
                if default:
                    return default
                raise ValueError()

        except Exception:
            raise KeyError((key,))

        # default, nothing
        raise KeyError((key,))


class TemplateResolver:
    class Result(Dict):
        def __init__(self, resolver: "TemplateResolver", template: dict):
            super().__init__()
            self._resolver = resolver
            self.update(template)

        @property
        def resolver(self):
            return self._resolver

        def set_dict(self, d: dict):
            d = deepcopy(d)
            self.clear()
            self.update(d)

    def __init__(self, task_session, task_id: str, queue_id: str, queue_name: str, worker_id: str, task_info=None):
        self._task_session = task_session
        self.queue_id = queue_id
        self.queue_name = queue_name
        self.task_id = task_id
        self.worker_id = worker_id
        self.user_vaults = LazyDict("user vaults", self._load_user_vaults, self._error)
        self.user_info = LazyDict("user info", self._load_user_info, error_cb=self._error)
        self.task_info = task_info if task_info else LazyDict("task info", self._load_task_info, self._error)
        self.provider_info = LazyDict("provider info", self._load_provider_info, self._error)

    @property
    def task_session(self):
        if not self._task_session:
            raise Exception(
                "Task session was not provided, probably due to the agent running with no use-owner-token option"
            )
        return self._task_session

    def _error(self, err):
        log.error(err)
        send_logs_to_task(
            worker_id=self.worker_id,
            task_id=self.task_id,
            lines=["ERROR: " + err],
            level="ERROR",
            session=self.task_session
        )

    def _load_user_info(self):
        return self.task_session.send_request("users", "get_current_user").json()["data"]["user"]

    def _load_user_vaults(self):
        self.task_session.load_vaults()
        return self.task_session.config.resolve_override_configs() or {}

    def _load_task_info(self):
        result = self.task_session.send_request(
            service="tasks",
            action="get_all",
            version='2.20',
            method=Request.def_method,
            json={"id": [self.task_id], "search_hidden": True}
        )
        # we should not fail here
        return result.json().get("data", {}).get("tasks", [])[0] or {}

    def _load_provider_info(self):
        token_dict = self.task_session.get_decoded_token(self.task_session.token, verify=False)
        return deepcopy(token_dict.get("providers_info") or {})

    def resolve_string_from_template(self, template: str) -> str:
        tmpl = CustomTemplate(template)
        return tmpl.default_custom_substitute(
            queue_id=self.queue_id,
            queue_name=self.queue_name,
            worker_id=self.worker_id,
            task_info=self.task_info,
            provider_info=None,
            config=self.task_session.config,
            user_vaults=self.user_vaults,
            user_info=self.user_info,
        )

    def resolve_from_template(self, template: str) -> Result:
        resolved_template_str = self.resolve_string_from_template(template)
        template = yaml.safe_load(resolved_template_str)
        return self.Result(self, template)
