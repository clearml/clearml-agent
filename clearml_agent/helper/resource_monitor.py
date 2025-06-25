from __future__ import unicode_literals, division

import logging
import re
import os
import shlex
from collections import deque
from itertools import starmap
from threading import Thread, Event
from time import time
from typing import Sequence, List, Union, Dict, Optional

from .._vendor import attr
import psutil
from .._vendor.pathlib2 import Path

from clearml_agent.definitions import ENV_WORKER_TAGS, ENV_GPU_FRACTIONS
from clearml_agent.session import Session

try:
    from .gpu import gpustat
except ImportError:
    gpustat = None

log = logging.getLogger(__name__)


class BytesSizes(object):
    @staticmethod
    def kilobytes(x):
        # type: (float) -> float
        return x / 1024

    @staticmethod
    def megabytes(x):
        # type: (float) -> float
        return x / (1024*1024)

    @staticmethod
    def gigabytes(x):
        # type: (float) -> float
        return x / (1024*1024*1024)


class ResourceMonitor(object):
    @attr.s
    class StatusReport(object):
        task = attr.ib(default=None, type=str)
        queue = attr.ib(default=None, type=str)
        queues = attr.ib(default=None, type=Sequence[str])

        def to_dict(self):
            return {
                key: value
                for key, value in attr.asdict(self).items()
                if value is not None
            }

    @attr.s
    class ClusterReport:
        cluster_key = attr.ib(type=str)
        max_gpus = attr.ib(type=int, default=None)
        max_workers = attr.ib(type=int, default=None)
        max_cpus = attr.ib(type=int, default=None)
        resource_groups = attr.ib(type=Sequence[str], factory=list)

    def __init__(
        self,
        session,  # type: Session
        worker_id,  # type: ResourceMonitor.StatusReport,
        sample_frequency_per_sec=2.0,
        report_frequency_sec=30.0,
        first_report_sec=None,
        worker_tags=None
    ):
        self.session = session
        self.queue = deque(maxlen=1)
        self.queue.appendleft(self.StatusReport())
        self._worker_id = worker_id
        self._sample_frequency = sample_frequency_per_sec
        self._report_frequency = report_frequency_sec
        self._first_report_sec = first_report_sec or report_frequency_sec
        self._num_readouts = 0
        self._readouts = {}
        self._previous_readouts = {}
        self._previous_readouts_ts = time()
        self._thread = None
        self._exit_event = Event()
        self._gpustat_fail = 0
        self._gpustat = gpustat
        self._active_gpus = None
        self._default_gpu_utilization = session.config.get("agent.resource_monitoring.default_gpu_utilization", 100)
        # allow default_gpu_utilization as null in the config, in which case we don't log anything
        if self._default_gpu_utilization is not None:
            self._default_gpu_utilization = int(self._default_gpu_utilization)
        self._gpu_utilization_warning_sent = False
        self._disk_use_path = str(session.config.get("agent.resource_monitoring.disk_use_path", None) or Path.home())
        self._fractions_handler = GpuFractionsHandler() if session.feature_set != "basic" else None
        if not worker_tags and ENV_WORKER_TAGS.get():
            worker_tags = shlex.split(ENV_WORKER_TAGS.get())
        self._worker_tags = worker_tags
        if Session.get_nvidia_visible_env() == 'none':
            # NVIDIA_VISIBLE_DEVICES set to none, marks cpu_only flag
            # active_gpus == False means no GPU reporting
            self._active_gpus = False
        elif not self._gpustat:
            log.warning('ClearML-Agent Resource Monitor: GPU monitoring is not available')
        else:
            # None means no filtering, report all gpus
            self._active_gpus = None
            # noinspection PyBroadException
            try:
                active_gpus = Session.get_nvidia_visible_env()
                # None means no filtering, report all gpus
                if active_gpus and active_gpus != "all":
                    if os.path.isdir(active_gpus):
                        try:
                            self._active_gpus = os.listdir(active_gpus)
                        except OSError as e:
                            log.warning(
                                "Failed listing {}: {}".format(active_gpus, e)
                            )
                    else:
                        self._active_gpus = [g.strip() for g in active_gpus.split(",")]
            except Exception:
                pass
        self._cluster_report_interval_sec = int(session.config.get(
            "agent.resource_monitoring.cluster_report_interval_sec", 60
        ))
        self._cluster_report = None

    def set_report(self, report):
        # type: (ResourceMonitor.StatusReport) -> ()
        if report is not None:
            self.queue.appendleft(report)

    def get_report(self):
        # type: () -> ResourceMonitor.StatusReport
        return self.queue[0]

    def start(self):
        self._exit_event.clear()
        self._thread = Thread(target=self._daemon)
        self._thread.daemon = True
        self._thread.start()
        return self

    def stop(self):
        self._exit_event.set()
        self.send_report()

    def send_report(self, stats=None):
        report = dict(
            machine_stats=stats,
            timestamp=(int(time()) * 1000),
            worker=self._worker_id,
            tags=self._worker_tags,
            **self.get_report().to_dict()
        )
        log.debug("sending report: %s", report)

        # noinspection PyBroadException
        try:
            self.session.get(service="workers", action="status_report", **report)
        except Exception:
            log.warning("Failed sending report: %s", report)
            return False
        return True

    def send_cluster_report(self) -> bool:
        if not self.session.feature_set == "basic":
            return False

        # noinspection PyBroadException
        try:
            properties = {
                "max_cpus": self._cluster_report.max_cpus,
                "max_gpus": self._cluster_report.max_gpus,
                "max_workers": self._cluster_report.max_workers,
            }
            payload = {
                "key": self._cluster_report.cluster_key,
                "timestamp": int(time() * 1000),
                "timeout": int(self._cluster_report_interval_sec * 2),
                # "resource_groups": self._cluster_report.resource_groups,  # yet to be supported
                "properties": {k: v for k, v in properties.items() if v is not None},
            }
            self.session.post(service="workers", action="cluster_report", **payload)
        except Exception as ex:
            log.warning("Failed sending cluster report: %s", ex)
            return False
        return True

    def setup_cluster_report(self, available_gpus, gpu_queues, worker_id=None, cluster_key=None, resource_groups=None):
        # type: (List[int], Dict[str, int], Optional[str], Optional[str], Optional[List[str]]) -> ()
        """
        Set up a cluster report for the enterprise server dashboard feature.
        If a worker_id is provided, cluster_key and resource_groups are inferred from it.
        """
        if self.session.feature_set == "basic":
            return

        if not worker_id and not cluster_key:
            print("Error: cannot set up dashboard reporting - worker_id or cluster key are required")
            return

        # noinspection PyBroadException
        try:
            if not cluster_key:
                worker_id_parts = worker_id.split(":")
                if len(worker_id_parts) < 3:
                    cluster_key = self.session.config.get("agent.resource_dashboard.default_cluster_name", "onprem")
                    resource_group = ":".join((cluster_key, worker_id_parts[0]))
                    print(
                        'WARNING: your worker ID "{}" is not suitable for proper resource dashboard reporting, please '
                        'set up agent.worker_name to be at least two colon-separated parts (i.e. "<category>:<name>"). '
                        'Using "{}" as the resource dashboard category and "{}" as the resource group.'.format(
                            worker_id, cluster_key, resource_group
                        )
                    )
                else:
                    cluster_key = worker_id_parts[0]
                    resource_group = ":".join((worker_id_parts[:2]))

                resource_groups = [resource_group]

            self._cluster_report = ResourceMonitor.ClusterReport(
                cluster_key=cluster_key,
                max_gpus=len(available_gpus),
                max_workers=len(available_gpus) // min(x for x, _ in gpu_queues.values()),
                resource_groups=resource_groups
            )

            self.send_cluster_report()
        except Exception as ex:
            print("Error: failed setting cluster report: {}".format(ex))

    def _daemon(self):
        last_cluster_report = 0
        seconds_since_started = 0
        reported = 0
        try:
            while True:
                last_report = time()
                current_report_frequency = (
                    self._report_frequency if reported != 0 else self._first_report_sec
                )
                while (time() - last_report) < current_report_frequency:
                    # wait for self._sample_frequency seconds, if event set quit
                    if self._exit_event.wait(1 / self._sample_frequency):
                        return
                    # noinspection PyBroadException
                    try:
                        self._update_readouts()
                    except Exception as ex:
                        log.error("failed getting machine stats: %s", report_error(ex))
                        self._failure()

                seconds_since_started += int(round(time() - last_report))
                # check if we do not report any metric (so it means the last iteration will not be changed)

                # if we do not have last_iteration, we just use seconds as iteration

                # start reporting only when we figured out, if this is seconds based, or iterations based
                average_readouts = self._get_average_readouts()
                stats = {
                    # 3 points after the dot
                    key: round(value, 3) if isinstance(value, float) else [round(v, 3) for v in value]
                    for key, value in average_readouts.items()
                }

                # send actual report
                if self.send_report(stats):
                    # clear readouts if this is update was sent
                    self._clear_readouts()

                # count reported iterations
                reported += 1

                if (
                    self._cluster_report and
                    self._cluster_report_interval_sec
                    and time() - last_cluster_report > self._cluster_report_interval_sec
                ):
                    if self.send_cluster_report():
                        last_cluster_report = time()

        except Exception as ex:
            log.exception("Error reporting monitoring info: %s", str(ex))

    def _update_readouts(self):
        readouts = self._machine_stats()
        elapsed = time() - self._previous_readouts_ts
        self._previous_readouts_ts = time()

        def fix(k, v):
            if k.endswith("_mbs"):
                v = (v - self._previous_readouts.get(k, v)) / elapsed

            if v is None:
                v = 0
            return k, self._readouts.get(k, 0) + v

        self._readouts.update(starmap(fix, readouts.items()))
        self._num_readouts += 1
        self._previous_readouts = readouts

    def _get_num_readouts(self):
        return self._num_readouts

    def _get_average_readouts(self):
        def create_general_key(old_key):
            """
            Create key for backend payload
            :param old_key: old stats key
            :type old_key: str
            :return: new key for sending stats
            :rtype: str
            """
            key_parts = old_key.rpartition("_")
            return "{}_*".format(key_parts[0] if old_key.startswith("gpu") else old_key)

        ret = {}
        # make sure the gpu/cpu stats are always ordered in the accumulated values list (general_key)
        ordered_keys = sorted(self._readouts.keys())
        for k in ordered_keys:
            v = self._readouts[k]
            stat_key = self.BACKEND_STAT_MAP.get(k)
            if stat_key:
                ret[stat_key] = v / self._num_readouts
            else:
                general_key = create_general_key(k)
                general_key = self.BACKEND_STAT_MAP.get(general_key)
                if general_key:
                    ret.setdefault(general_key, []).append(v / self._num_readouts)
                else:
                    pass  # log.debug("Cannot find key {}".format(k))
        return ret

    def _clear_readouts(self):
        self._readouts = {}
        self._num_readouts = 0

    def _machine_stats(self):
        """
        :return: machine stats dictionary, all values expressed in megabytes
        """
        cpu_usage = psutil.cpu_percent(percpu=True)
        stats = {"cpu_usage": sum(cpu_usage) / len(cpu_usage)}

        virtual_memory = psutil.virtual_memory()
        stats["memory_used"] = BytesSizes.megabytes(virtual_memory.used)
        stats["memory_free"] = BytesSizes.megabytes(virtual_memory.available)
        disk_use_percentage = psutil.disk_usage(self._disk_use_path).percent
        stats["disk_free_percent"] = 100 - disk_use_percentage
        sensor_stat = (
            psutil.sensors_temperatures()
            if hasattr(psutil, "sensors_temperatures")
            else {}
        )
        if "coretemp" in sensor_stat and len(sensor_stat["coretemp"]):
            stats["cpu_temperature"] = max([t.current for t in sensor_stat["coretemp"]])

        # update cached measurements
        net_stats = psutil.net_io_counters()
        stats["network_tx_mbs"] = BytesSizes.megabytes(net_stats.bytes_sent)
        stats["network_rx_mbs"] = BytesSizes.megabytes(net_stats.bytes_recv)
        io_stats = psutil.disk_io_counters()
        stats["io_read_mbs"] = BytesSizes.megabytes(io_stats.read_bytes)
        stats["io_write_mbs"] = BytesSizes.megabytes(io_stats.write_bytes)

        # check if we need to monitor gpus and if we can access the gpu statistics
        if self._active_gpus is not False and self._gpustat:
            try:
                gpu_stat = self._gpustat.new_query()
                report_index = 0
                for i, g in enumerate(gpu_stat.gpus):
                    # only monitor the active gpu's, if none were selected, monitor everything
                    if self._active_gpus:
                        uuid = getattr(g, "uuid", None)
                        mig_uuid = getattr(g, "mig_uuid", None)
                        if (
                            str(g.index) not in self._active_gpus
                            and (not uuid or uuid not in self._active_gpus)
                            and (not mig_uuid or mig_uuid not in self._active_gpus)
                        ):
                            continue
                    stats["gpu_temperature_{}".format(report_index)] = g["temperature.gpu"]

                    if g["utilization.gpu"] is not None:
                        stats["gpu_utilization_{}".format(report_index)] = g["utilization.gpu"]
                    elif self._default_gpu_utilization is not None:
                        stats["gpu_utilization_{}".format(report_index)] = self._default_gpu_utilization
                        if getattr(g, "mig_index", None) is None and not self._gpu_utilization_warning_sent:
                            # this shouldn't happen for non-MIGs, warn the user about it
                            log.error("Failed fetching GPU utilization")
                            self._gpu_utilization_warning_sent = True

                    stats["gpu_mem_usage_{}".format(report_index)] = (
                        100.0 * g["memory.used"] / g["memory.total"]
                    )
                    # already in MBs
                    stats["gpu_mem_free_{}".format(report_index)] = (
                        g["memory.total"] - g["memory.used"]
                    )

                    stats["gpu_mem_used_{}".format(report_index)] = g["memory.used"] or 0

                    if self._fractions_handler:
                        fractions = self._fractions_handler.fractions
                        stats["gpu_fraction_{}".format(report_index)] = \
                            (fractions[i] if i < len(fractions) else fractions[-1]) if fractions else 1.0
                    report_index += 1

            except Exception as ex:
                # something happened and we can't use gpu stats,
                log.error("failed getting machine stats: %s", report_error(ex))
                self._failure()

        return stats

    def _failure(self):
        self._gpustat_fail += 1
        if self._gpustat_fail >= 3:
            log.error(
                "GPU monitoring failed getting GPU reading, switching off GPU monitoring"
            )
            self._gpustat = None

    BACKEND_STAT_MAP = {
        "cpu_usage_*": "cpu_usage",
        "cpu_temperature_*": "cpu_temperature",
        "disk_free_percent": "disk_free_home",
        "io_read_mbs": "disk_read",
        "io_write_mbs": "disk_write",
        "network_tx_mbs": "network_tx",
        "network_rx_mbs": "network_rx",
        "memory_free": "memory_free",
        "memory_used": "memory_used",
        "gpu_temperature_*": "gpu_temperature",
        "gpu_mem_used_*": "gpu_memory_used",
        "gpu_mem_free_*": "gpu_memory_free",
        "gpu_utilization_*": "gpu_usage",
        "gpu_fraction_*": "gpu_fraction"
    }


class GpuFractionsHandler:
    _number_re = re.compile(r"^clear\.ml/fraction(-\d+)?$")
    _mig_re = re.compile(r"^nvidia\.com/mig-(?P<compute>[0-9]+)g\.(?P<memory>[0-9]+)gb$")
    _frac_gpu_injector_re = re.compile(r"^clearml-injector/fraction$")

    _gpu_name_to_memory_gb = {
        "NVIDIA-A30": 24,
        "NVIDIA-A100-40GB-PCIE": 40,
        "NVIDIA-A100-SXM4-40GB": 40,
        "NVIDIA-A100-SXM4-80GB": 79,
        "NVIDIA-A100-80GB-PCIE": 79,
        "NVIDIA-A100-PCIE-40GB": 40,
        "NVIDIA-H100-80GB-HBM3": 80,
        "NVIDIA-H100-PCIE": 80,
        "NVIDIA-H100-SXM5-94GB": 94,
        "NVIDIA-H100-SXM5-96GB": 96,
        "NVIDIA-H100-PCIE-94GB": 94,
        "NVIDIA-H100-PCIE-96GB": 96,
        "NVIDIA-H100-SXM5-64GB": 64,
        "NVIDIA-H100-SXM5-80GB": 80,
        "NVIDIA-H800-PCIE-94GB": 94,
        "NVIDIA-H800-PCIE-80GB": 80,
        "NVIDIA-L40S": 48,
        "NVIDIA-L40": 48,
        "NVIDIA-L4": 48,
    }

    def __init__(self):
        self._total_memory_gb = [
            (self._gpu_name_to_memory_gb.get(name.upper().replace(" ", "-")) or
             self._gpu_name_to_memory_gb.get("NVIDIA-"+name.upper().replace(" ", "-"), 0))
            for name in (self._get_gpu_names() or [])
        ]
        self._fractions = self._get_fractions()

    @property
    def fractions(self) -> List[float]:
        return self._fractions

    def _get_fractions(self) -> List[float]:
        if not self._total_memory_gb:
            # Can't compute
            return [1.0]

        fractions = (ENV_GPU_FRACTIONS.get() or "").strip()
        if not fractions:
            # No fractions
            return [1.0]

        decoded_fractions = self.decode_fractions(fractions)

        if isinstance(decoded_fractions, list):
            return decoded_fractions

        totals = []
        for i, (fraction, count) in enumerate(decoded_fractions.items()):
            m = self._mig_re.match(fraction)
            if not m:
                continue
            try:
                total_gb = self._total_memory_gb[i] if i < len(self._total_memory_gb) else self._total_memory_gb[-1]
                if not total_gb:
                    continue
                totals.append((int(m.group("memory")) * count) / total_gb)
            except ValueError:
                pass

        if not totals:
            log.warning("Fractions count is empty for {}".format(fractions))
            return [1.0]

        return totals

    @classmethod
    def extract_custom_limits(cls, limits: dict):
        for k, v in list((limits or {}).items()):
            if cls._number_re.match(k):
                limits.pop(k, None)

    @classmethod
    def get_simple_fractions_total(cls, limits: dict) -> float:
        try:
            if any(cls._number_re.match(x) for x in limits):
                return sum(float(v) for k, v in limits.items() if cls._number_re.match(k))
        except Exception as ex:
            log.error("Failed summing up fractions from {}: {}".format(limits, ex))
        return 0

    @classmethod
    def encode_fractions(cls, limits: dict, annotations: dict) -> str:
        if limits:
            if any(cls._number_re.match(x) for x in (limits or {})):
                return ",".join(str(v) for k, v in sorted(limits.items()) if cls._number_re.match(k))
            return ",".join(("{}:{}".format(k, v) for k, v in (limits or {}).items() if cls._mig_re.match(k)))
        elif annotations:
            if any(cls._frac_gpu_injector_re.match(x) for x in (annotations or {})):
                return ",".join(str(v) for k, v in sorted(annotations.items()) if cls._frac_gpu_injector_re.match(k))

    @staticmethod
    def decode_fractions(fractions: str) -> Union[List[float], Dict[str, int]]:
        try:
            items = [f.strip() for f in fractions.strip().split(",")]
            tuples = [(k.strip(), v.strip()) for k, v in (f.partition(":")[::2] for f in items)]
            if all(not v for _, v in tuples):
                # comma-separated float fractions
                return [float(k) for k, _ in tuples]
            # comma-separated slice:count items
            return {
                k.strip(): int(v.strip())
                for k, v in tuples
            }
        except Exception as ex:
            log.error("Failed decoding GPU fractions '{}': {}".format(fractions, ex))
        return {}

    @staticmethod
    def _get_gpu_names():
        # noinspection PyBroadException
        try:
            gpus = gpustat.new_query().gpus
            names = [g["name"] for g in gpus]

            print("GPU names: {}".format(names))

            return names
        except Exception as ex:
            log.error("Failed getting GPU names: {}".format(ex))


def report_error(ex):
    return "{}: {}".format(type(ex).__name__, ex)
