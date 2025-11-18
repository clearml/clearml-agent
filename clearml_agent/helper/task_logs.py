from typing import List


def send_logs_to_task(worker_id: str, task_id: str, lines: List[str], session, level="DEBUG"):
    """
    Send output lines as log events to backend
    :param worker_id: Worker ID
    :param task_id: ID of task to send logs for
    :type task_id: Text
    :param lines: lines to send
    :type lines: [Text]
    :param str level: log level, default DEBUG
    :param session: Worker Session to send logs to
    :return: number of lines sent
    :rtype: int
    """
    # import locally to avoid circular imports
    from clearml_agent.helper.console import print_text
    from clearml_agent.commands.events import Events

    if not lines:
        return 0
    print_text("".join(lines), newline=False)

    # remove backspaces from the text log, they look bad.
    for i, l in enumerate(lines):
        lines[i] = l.replace('\x08', '')

    events_service = Events(session.config)
    try:
        events_service.send_log_events(
            worker_id, task_id=task_id, lines=lines, level=level, session=session
        )
        return len(lines)
    except Exception as e:
        print("\n### Error sending log: %s ###\n" % e)
        # revert number of sent lines (we will try next time)
        return 0
