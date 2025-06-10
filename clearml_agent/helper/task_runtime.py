from typing import Optional

from ..backend_api.session import Request


class TaskRuntime(object):

    def __init__(self, session):
        self._session = session

    def get_task_runtime(self, task_id) -> Optional[dict]:
        try:
            res = self._session.send_request(
                service='tasks', action='get_by_id', method=Request.def_method,
                json={"task": task_id, "only_fields": ["runtime"]},
            )
            if not res.ok:
                raise ValueError(f"request returned {res.status_code}")
            data = res.json().get("data")
            if not data or "task" not in data:
                raise ValueError("empty data in result")
            return data["task"].get("runtime", {})
        except Exception as ex:
            print(f"ERROR: Failed getting runtime properties for task {task_id}: {ex}")

    def update_task_runtime(self, task_id: str, runtime: dict) -> bool:
        task_runtime = self.get_task_runtime(task_id) or {}
        task_runtime.update(runtime)

        try:
            res = self._session.send_request(
                service='tasks', action='edit', method=Request.def_method,
                json={
                    "task": task_id, "force": True, "runtime": task_runtime
                },
            )
            if not res.ok:
                raise Exception("failed setting runtime property")
            return True
        except Exception as ex:
            print("WARNING: failed setting custom runtime properties for task '{}': {}".format(task_id, ex))

        return False
