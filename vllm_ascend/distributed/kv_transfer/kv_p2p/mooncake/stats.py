# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Transfer statistics and Prometheus metrics for Mooncake."""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.v1.metrics.utils import create_metric_per_engine


@dataclass
class MooncakeKVConnectorStats(KVConnectorStats):
    """D2D operation stats and aggregated Decode receive outcomes."""

    _DATA_KEYS = (
        "transfer_duration",
        "bytes_transferred",
        "num_failed_transfers",
        "receive_wait_duration",
        "num_failed_receives",
    )

    def __post_init__(self) -> None:
        if not self.data:
            self.reset()
        else:
            for key in self._DATA_KEYS:
                self.data.setdefault(key, [])

    def reset(self) -> None:
        # Values must remain serializable because worker stats are aggregated
        # outside of the worker process.
        self.data: dict[str, list[float | int]] = {key: [] for key in self._DATA_KEYS}

    def record_transfer(self, transfer_duration: float, bytes_transferred: int) -> None:
        """Record one completed D2D transfer.

        ``transfer_duration`` is expressed in seconds.
        """
        self.data["transfer_duration"].append(transfer_duration)
        self.data["bytes_transferred"].append(bytes_transferred)

    def record_failed_transfer(self) -> None:
        self.data["num_failed_transfers"].append(1)

    def record_receive(self, duration: float, failed: bool) -> None:
        if failed:
            self.data["num_failed_receives"].append(1)
        else:
            self.data["receive_wait_duration"].append(duration)

    def clone_and_reset(self) -> "MooncakeKVConnectorStats":
        previous = copy.copy(self)
        self.reset()
        return previous

    def is_empty(self) -> bool:
        return not any(self.data.values())

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not other.is_empty():
            for key, values in other.data.items():
                accumulator = self.data[key]
                assert isinstance(accumulator, list)
                accumulator.extend(values)
        return self

    def reduce(self) -> dict[str, int | float]:
        if self.num_successful_transfers == 0:
            return {
                "Num successful transfers": 0,
                "Num failed transfers": len(self.data["num_failed_transfers"]),
                "Avg xfer time (ms)": 0,
                "P90 xfer time (ms)": 0,
                "Avg MB per transfer": 0,
                "Throughput (MB/s)": 0,
            }

        durations = np.asarray(self.data["transfer_duration"])
        megabytes = np.asarray(self.data["bytes_transferred"]) / 2**20
        total_duration = durations.sum()
        throughput = megabytes.sum() / total_duration if total_duration > 0 else 0

        return {
            "Num successful transfers": self.num_successful_transfers,
            "Num failed transfers": len(self.data["num_failed_transfers"]),
            "Avg xfer time (ms)": round(durations.mean() * 1e3, 3),
            "P90 xfer time (ms)": round(np.percentile(durations, 90).item() * 1e3, 3),
            "Avg MB per transfer": round(megabytes.mean(), 3),
            "Throughput (MB/s)": round(float(throughput), 3),
        }

    @property
    def num_successful_transfers(self) -> int:
        return len(self.data["transfer_duration"])


class MooncakeTransferRecorder:
    """Collect transport calls from executor threads and drain from the model runner."""

    def __init__(self) -> None:
        self.stats = MooncakeKVConnectorStats()
        self._lock = Lock()

    def transfer(
        self,
        operation: Callable[[str, list[int], list[int], list[int]], int],
        session: str,
        src: list[int],
        dst: list[int],
        lengths: list[int],
    ) -> tuple[int, float, int]:
        started = time.perf_counter()
        try:
            result = operation(session, src, dst, lengths)
        except Exception:
            with self._lock:
                self.stats.record_failed_transfer()
            raise

        elapsed = time.perf_counter() - started
        num_bytes = sum(lengths) if result >= 0 else 0
        with self._lock:
            if result < 0:
                self.stats.record_failed_transfer()
            else:
                self.stats.record_transfer(elapsed, num_bytes)
        return result, elapsed, num_bytes

    def drain(self) -> MooncakeKVConnectorStats | None:
        with self._lock:
            return None if self.stats.is_empty() else self.stats.clone_and_reset()


class MooncakePromMetrics(KVConnectorPromMetrics):
    """Prometheus metrics for Mooncake D2D transfers."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

        transfer_duration = self._histogram_cls(
            name="vllm:mooncake_xfer_time_seconds",
            documentation=("Histogram of Mooncake KV cache D2D transfer duration."),
            buckets=[
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.2,
                0.3,
                0.5,
                0.75,
                1.0,
                5.0,
            ],
            labelnames=labelnames,
        )
        self.transfer_duration = create_metric_per_engine(transfer_duration, self.per_engine_labelvalues)

        bytes_transferred = self._histogram_cls(
            name="vllm:mooncake_bytes_transferred",
            documentation=("Histogram of bytes transferred per Mooncake KV cache D2D transfer."),
            buckets=[2 ** (10 + index) for index in range(1, 25, 2)],
            labelnames=labelnames,
        )
        self.bytes_transferred = create_metric_per_engine(bytes_transferred, self.per_engine_labelvalues)

        failed_transfers = self._counter_cls(
            name="vllm:mooncake_num_failed_transfers",
            documentation="Number of failed Mooncake KV cache D2D transfers.",
            labelnames=labelnames,
        )
        self.failed_transfers = create_metric_per_engine(failed_transfers, self.per_engine_labelvalues)

        receive_wait = self._histogram_cls(
            name="vllm:mooncake_receive_wait_seconds",
            documentation="Time from Decode allocation for remote KV to successful aggregated receive.",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 60.0],
            labelnames=labelnames,
        )
        self.receive_wait = create_metric_per_engine(receive_wait, self.per_engine_labelvalues)
        failed_receives = self._counter_cls(
            name="vllm:mooncake_num_failed_receives",
            documentation="Number of real Decode requests with an aggregated Mooncake receive failure.",
            labelnames=labelnames,
        )
        self.failed_receives = create_metric_per_engine(failed_receives, self.per_engine_labelvalues)

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for duration in transfer_stats_data["transfer_duration"]:
            self.transfer_duration[engine_idx].observe(duration)
        for num_bytes in transfer_stats_data["bytes_transferred"]:
            self.bytes_transferred[engine_idx].observe(num_bytes)
        for failure in transfer_stats_data["num_failed_transfers"]:
            self.failed_transfers[engine_idx].inc(failure)
        for duration in transfer_stats_data.get("receive_wait_duration", ()):
            self.receive_wait[engine_idx].observe(duration)
        if failures := transfer_stats_data.get("num_failed_receives"):
            self.failed_receives[engine_idx].inc(sum(failures))
