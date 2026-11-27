# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV transfer metrics for Prefill/Decode (PD) disaggregation.

Prefill and Decode run on separate instances and the Decode side pulls the
computed KV cache through a KV connector. This module only hooks existing
connector/scheduler methods:

- receives completed per poll (``xfer_finished_total``);
- blocks that failed to load (``invalid_blocks_total``);
- requests still waiting for remote KV per step
  (``remote_kv_waiting_requests``).

Mooncake V2 counts real receives after worker aggregation. Its waiting gauge
tracks real receives from Decode allocation until completion.
"""

import re
from typing import Any

from vllm_ascend.observability.ms_metrics.handlers.common import (
    METRIC_LEVEL_DETAIL,
    MetricSpec,
    MetricType,
    metric_enabled,
    record_observation,
    safe_recording,
)

_KEY = "kv_transfer"
_CONNECTOR_LABEL = "connector"

_FINISHED_SPEC = MetricSpec(
    "kv:xfer_finished_total",
    MetricType.COUNTER,
    (_CONNECTOR_LABEL,),
)
_INVALID_BLOCKS_SPEC = MetricSpec(
    "kv:invalid_blocks_total",
    MetricType.COUNTER,
    (_CONNECTOR_LABEL,),
)
_REMOTE_WAITING_SPEC = MetricSpec(
    "kv:remote_kv_waiting_requests",
    MetricType.GAUGE,
    (_CONNECTOR_LABEL,),
    level=METRIC_LEVEL_DETAIL,
)
_SUCCEEDED_SPEC = MetricSpec("kv:xfer_succeeded_total", MetricType.COUNTER, (_CONNECTOR_LABEL,))
_FAILED_SPEC = MetricSpec("kv:xfer_failed_total", MetricType.COUNTER, (_CONNECTOR_LABEL,))
_RECEIVE_WAIT_SPEC = MetricSpec(
    "kv:remote_kv_wait:duration_ms",
    MetricType.HISTOGRAM,
    (_CONNECTOR_LABEL,),
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 15000, 60000),
)
_SPECS = (
    _FINISHED_SPEC,
    _INVALID_BLOCKS_SPEC,
    _REMOTE_WAITING_SPEC,
    _SUCCEEDED_SPEC,
    _FAILED_SPEC,
    _RECEIVE_WAIT_SPEC,
)

_recorders: dict[str, Any] = {}

# Stable connector label values. The same class name (e.g. MooncakeConnector)
# is reused by different backend modules, so the module wins when it carries
# the disambiguating information.
_MODULE_LABELS = (
    ("mooncake_hybrid_connector", "mooncake_hybrid"),
    ("mooncake_layerwise_connector", "mooncake_layerwise"),
)
_CLASS_LABELS = {
    "MooncakeConnector": "mooncake",
    "MooncakeConnectorScheduler": "mooncake",
    "MooncakePullConnectorScheduler": "mooncake",
    "MooncakeLayerwiseConnector": "mooncake_layerwise",
    "MooncakeLayerwiseConnectorScheduler": "mooncake_layerwise",
    "MooncakeHybridConnector": "mooncake_hybrid",
    "MooncakePullConnector": "mooncake_pull",
    "MooncakePushConnector": "mooncake_push",
    "MooncakeConnectorV2": "mooncake_v2",
    "AscendStoreConnector": "ascend_store",
    "BaseKVPoolScheduler": "ascend_store",
    "KVPoolScheduler": "ascend_store",
    "LayerwiseKVPoolScheduler": "ascend_store",
    "SfaRemoteD2HConnector": "sfa",
    "SFAPDRD2HScheduler": "sfa",
    "SFAPDRD2HConsumerWorker": "sfa",
    "UCMConnectorV1": "ucm",
}
_CLASS_PREFIX_LABELS = (
    ("SFA", "sfa"),
    ("UCM", "ucm"),
)
_WAITING_ATTRIBUTES = ("_reqs_need_recv", "_reqs_recv_info", "_async_loading_req_ids")

_CAMEL_CASE_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(name: str) -> str:
    return _CAMEL_CASE_BOUNDARY.sub("_", name).lower()


def _connector_label(obj: Any) -> str:
    class_name = type(obj).__name__
    module_name = type(obj).__module__
    for marker, label in _MODULE_LABELS:
        if marker in module_name:
            return label
    label = _CLASS_LABELS.get(class_name)
    if label is not None:
        return label
    for prefix, prefix_label in _CLASS_PREFIX_LABELS:
        if class_name.startswith(prefix):
            return prefix_label
    return _camel_to_snake(class_name)


def _record_finished(count: int, connector: Any) -> None:
    if not count:
        return
    record_observation(
        _recorders,
        _KEY,
        _SPECS,
        _FINISHED_SPEC,
        float(count),
        {_CONNECTOR_LABEL: _connector_label(connector)},
        "kv_transfer",
    )


def kv_transfer_finished_handler(original_func, connector, *args, **kwargs):
    """Count finished receives from ``KVConnectorTransferResults``.

    Used by connectors that implement the V2 ``get_transfer_results`` API.
    """
    result = original_func(connector, *args, **kwargs)

    finished_recving = getattr(result, "finished_recving", None)
    _record_finished(len(finished_recving) if finished_recving else 0, connector)
    return result


def mooncake_receive_stats_handler(original_func, connector, *args, **kwargs):
    """Record the outcomes drained by the scheduler after cross-rank aggregation."""
    stats = original_func(connector, *args, **kwargs)
    if stats is not None:
        with safe_recording(_KEY):
            durations = stats.data["receive_wait_duration"]
            failed = sum(stats.data["num_failed_receives"])
            if not durations and not failed:
                return stats
            _record_mooncake_waiting(connector)
            _record_finished(len(durations) + failed, connector)
            labels = {_CONNECTOR_LABEL: _connector_label(connector)}
            if durations:
                record_observation(_recorders, _KEY, _SPECS, _SUCCEEDED_SPEC, len(durations), labels, _KEY)
                for duration in durations:
                    record_observation(_recorders, _KEY, _SPECS, _RECEIVE_WAIT_SPEC, duration * 1000, labels, _KEY)
            if failed:
                record_observation(_recorders, _KEY, _SPECS, _FAILED_SPEC, failed, labels, _KEY)
    return stats


def _record_mooncake_waiting(connector: Any) -> None:
    if not metric_enabled(_REMOTE_WAITING_SPEC):
        return
    with safe_recording(_KEY):
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _REMOTE_WAITING_SPEC,
            len(connector._receive_wait_started),
            # The gauge describes one queue across scheduler phases; an
            # implicit phase label would leave stale values in old series.
            {_CONNECTOR_LABEL: _connector_label(connector), "phase": "all"},
            _KEY,
        )


def mooncake_waiting_handler(original_func, connector, *args, **kwargs):
    result = original_func(connector, *args, **kwargs)
    _record_mooncake_waiting(connector)
    return result


def kv_transfer_invalid_blocks_handler(original_func, connector, *args, **kwargs):
    """Count blocks that failed to load from the remote KV cache.

    ``get_block_ids_with_load_errors`` returns the pending block ids, so the
    handler only inspects the returned set.
    """
    result = original_func(connector, *args, **kwargs)

    failed_blocks = len(result) if result else 0
    if failed_blocks:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _INVALID_BLOCKS_SPEC,
            float(failed_blocks),
            {_CONNECTOR_LABEL: _connector_label(connector)},
            "kv_transfer",
        )
    return result


def _waiting_request_count(scheduler: Any) -> int | None:
    """Return the number of requests still waiting for remote KV.

    ``_reqs_need_recv`` only holds requests queued for the current step: the
    connector clears it inside ``build_connector_meta`` when the READs are
    dispatched, while the transfer keeps running for several scheduler steps.
    Connectors that track in-flight registrations separately (for example
    ``_reqs_recv_info`` on the Mooncake pull scheduler) are unioned in, so a
    multi-step READ does not report zero until it actually completes.

    ``None`` means no known collection exists, which must not be recorded as a
    zero sample.
    """
    waiting: set[str] = set()
    seen_attribute = False
    for attribute in _WAITING_ATTRIBUTES:
        pending = getattr(scheduler, attribute, None)
        if pending is None:
            continue
        seen_attribute = True
        waiting.update(pending)
    return len(waiting) if seen_attribute else None


def kv_transfer_remote_waiting_handler(original_func, scheduler, *args, **kwargs):
    """Report per-step requests still waiting for remote KV.

    ``build_connector_meta`` drains the pending collection before returning,
    so the sample is taken before invoking the wrapped method. Zero is a valid
    sample and is recorded; a missing collection is not.

    The snapshot runs inside ``safe_recording`` and only when the level allows
    it: an unexpected attribute shape must not stop ``build_connector_meta``
    from running, and nothing is extracted when the metric is disabled.
    """
    waiting = None
    if metric_enabled(_REMOTE_WAITING_SPEC):
        with safe_recording(_KEY):
            waiting = _waiting_request_count(scheduler)
    result = original_func(scheduler, *args, **kwargs)

    if waiting is not None:
        record_observation(
            _recorders,
            _KEY,
            _SPECS,
            _REMOTE_WAITING_SPEC,
            float(waiting),
            {_CONNECTOR_LABEL: _connector_label(scheduler)},
            "kv_transfer",
        )
    return result
