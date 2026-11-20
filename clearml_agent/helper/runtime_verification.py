import re
from datetime import datetime, timedelta

from typing import List, Tuple, Optional

from clearml_agent.backend_config.defs import UptimeConf

DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
PATTERN = re.compile(r"^(?P<hours>[^\s]+)\s(?P<days>[^\s]+)")


def check_runtime(ranges_list, is_uptime=True):
    # type: (List[str], bool) -> bool
    for entry in ranges_list:

        days_list = get_days_list(entry)
        if not check_day(days_list):
            continue

        hours_list = get_hours_list(entry)
        if check_hour(hours_list):
            return is_uptime
    return not is_uptime


def check_hour(hours):
    # type: (List[str]) -> bool
    return datetime.now().hour in hours


def check_day(days):
    # type: (List[str]) -> bool
    return datetime.now().strftime("%a").upper() in days


def get_days_list(entry):
    # type: (str) -> List[str]
    days_intervals = PATTERN.match(entry)["days"].split(",")
    days_total = []
    for days in days_intervals:
        start, end = days.split("-") if "-" in days else (days, days)
        try:
            days_total.extend(
                [*range(DAYS.index(start.upper()), DAYS.index(end.upper()) + 1)]
            )
        except ValueError:
            print(
                "Warning: days interval '{}' is invalid, use intervals of the format <start>-<end>."
                " make sure to use the abbreviated format SUN-SAT".format(days)
            )
            continue
    return [DAYS[day] for day in days_total]


def get_hours_list(entry):
    # type: (str) -> List[str]
    hours_intervals = PATTERN.match(entry)["hours"].split(",")
    hours_total = []
    for hours in hours_intervals:
        start, end = get_start_end_hours(hours)
        if not (start and end):
            continue
        hours_total.extend([*range(start, end)])
    return hours_total


def get_start_end_hours(hours):
    # type: (str) -> Tuple[int, int]
    try:
        start, end = (
            tuple(map(int, hours.split("-"))) if "-" in hours else (int(hours), 24)
        )
    except Exception as ex:
        print(
            "Warning: hours interval '{}' is invalid, use intervals of the format <start>-<end> ({})".format(
                hours, ex
            )
        )
        start, end = (None, None)
    if end == 0:
        end = 24
    return start, end


def print_uptime_properties(
    ranges_list, queues_info, runtime_properties, is_uptime=True
):
    # type: (List[str], List[dict], List[dict], bool) -> None
    if ranges_list:
        uptime_string = ["Working hours {} configurations".format("uptime" if is_uptime else "downtime")]
        uptime_string.extend(get_uptime_string(entry) for entry in ranges_list)
    else:
        uptime_string = ["No uptime/downtime configurations found"]

    is_server_forced, server_string = get_runtime_properties_string(runtime_properties)
    is_queue_forced, queues_string = get_queues_tags_string(queues_info)

    res = list(
        filter(
            len,
            [
                "\n".join(uptime_string),
                "\nCurrently forced {}".format(is_queue_forced or is_server_forced)
                if is_queue_forced or is_server_forced
                else " ",
                server_string,
                queues_string,
            ],
        )
    )
    print("\n".join(res))


def get_uptime_string(entry):
    # type: (str) -> str
    res = []
    days_list = get_days_list(entry)
    hours_intervals = PATTERN.match(entry)["hours"].split(",")
    for hours in hours_intervals:
        start, end = get_start_end_hours(hours)
        if not (start and end):
            continue
        res.append(
            "  - {}:00-{}:59 on {}".format(start, end - 1, ' and '.join(days_list))
            if not (start == end)
            else ""
        )
    return "\n".join(res)


def get_runtime_properties_string(runtime_properties):
    # type: (Optional[List[dict]]) -> Tuple[Optional[str], str]
    server_string = []
    runtime_properties = runtime_properties or []
    force_flag = next(
        (prop for prop in runtime_properties if prop["key"] == UptimeConf.worker_key),
        None,
    )
    is_server_forced = None
    if force_flag:
        is_server_forced = force_flag["value"].upper()
        expiry_hour = (
            (datetime.now() + timedelta(seconds=force_flag["expiry"])).strftime("%H:%M")
            if force_flag["expiry"]
            else None
        )
        expires = " expires at {}".format(expiry_hour) if expiry_hour else ""
        server_string.append(
            "  - Server runtime property '{}: {}'{}".format(force_flag['key'], force_flag['value'], expires)
        )
    return is_server_forced, "\n".join(server_string)


def get_queues_tags_string(queues_info):
    # type: (List[dict]) -> Tuple[Optional[str], str]
    queues_string = []
    is_queue_forced = None
    for queue in queues_info:
        if "force_workers:off" in queue.get("tags", []):
            tag = "force_workers:off"
            is_queue_forced = is_queue_forced or "OFF"
        elif "force_workers:on" in queue.get("tags", []):
            tag = "force_workers:on"
            is_queue_forced = "ON"
        else:
            tag = None
        tagged = " (tagged '{}')'".format(tag) if tag else ""
        queues_string.append(
            "  - Listening to queue '{}'{}".format(queue.get('name', ''), tagged)
        )
    return is_queue_forced, "\n".join(queues_string)
