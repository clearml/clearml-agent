
class GetPodsError(Exception):
    pass


class GetJobsError(Exception):
    pass


class GetPodCountError(Exception):
    pass

class ReenqueueTaskException(Exception):
    # Lower priority queues wait until the task is launched. Set to False on the raised exception to keep serving
    # them meanwhile (tasks behind it in its own queue still wait for it)
    strict_priority = True

