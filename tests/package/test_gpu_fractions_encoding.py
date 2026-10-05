"""
k8s glue encoding side: which fraction gets injected into the task pod's CLEARML_AGENT_GPU_FRACTIONS.

Regression: a CFGI queue template that also sets cpu/memory limits used to encode "", so the pod never
got the env var and a 0.5 GPU task was reported on the Orchestration Dashboard as a full GPU.
"""
from clearml_agent.definitions import ENV_GPU_FRACTIONS
from clearml_agent.helper.resource_monitor import GpuFractionsHandler

_GPU_FRACTIONS_VAR = ENV_GPU_FRACTIONS.vars[0]  # "CLEARML_AGENT_GPU_FRACTIONS"
_CPU_MEM_LIMITS = {"cpu": "4", "memory": "32Gi"}


def test_cfgi_label_encoded_despite_cpu_memory_limits():
    labels = {"clearml-injector/fraction": "0.500"}
    assert GpuFractionsHandler.encode_fractions(limits=dict(_CPU_MEM_LIMITS), labels=labels, annotations={}) == "0.500"


def test_custom_annotation_encoded_despite_cpu_memory_limits():
    annotations = {"gpu-fraction": "0.25"}
    assert GpuFractionsHandler.encode_fractions(limits=dict(_CPU_MEM_LIMITS), labels={}, annotations=annotations) == "0.25"


def test_cfgi_label_encoded_without_limits():
    assert GpuFractionsHandler.encode_fractions(limits=None, labels={"clearml-injector/fraction": "0.5"}) == "0.500"


def test_cfgi_whole_gpus_capped_to_one():
    assert GpuFractionsHandler.encode_fractions(limits=dict(_CPU_MEM_LIMITS), labels={"clearml-injector/fraction": "2"}) == "1.000"


def test_limits_fractions_take_precedence_over_labels():
    limits = dict(_CPU_MEM_LIMITS, **{"clear.ml/fraction-1": "0.25"})
    labels = {"clearml-injector/fraction": "0.500"}
    assert GpuFractionsHandler.encode_fractions(limits=limits, labels=labels) == "0.25"


def test_mig_limits_take_precedence_over_labels():
    limits = dict(_CPU_MEM_LIMITS, **{"nvidia.com/mig-1g.10gb": "2"})
    labels = {"clearml-injector/fraction": "0.500"}
    assert GpuFractionsHandler.encode_fractions(limits=limits, labels=labels) == "nvidia.com/mig-1g.10gb:2"


def test_no_fraction_source_encodes_empty():
    assert GpuFractionsHandler.encode_fractions(limits=dict(_CPU_MEM_LIMITS), labels={"app": "x"}, annotations={}) == ""
    assert GpuFractionsHandler.encode_fractions(limits=None, labels=None, annotations=None) == ""


def test_encoded_cfgi_fraction_round_trips_to_reported_value(monkeypatch):
    """What the glue encodes is what the task pod's resource monitor reports."""
    encoded = GpuFractionsHandler.encode_fractions(
        limits=dict(_CPU_MEM_LIMITS), labels={"clearml-injector/fraction": "0.500"}, annotations={}
    )
    monkeypatch.setenv(_GPU_FRACTIONS_VAR, encoded)
    monkeypatch.setattr(GpuFractionsHandler, "_get_gpu_names", staticmethod(lambda: ["NVIDIA RTX A5000"]))
    assert GpuFractionsHandler().fractions == [0.5]
