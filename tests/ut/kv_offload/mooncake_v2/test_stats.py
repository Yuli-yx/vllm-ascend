# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.stats import (
    MooncakeKVConnectorStats,
    MooncakeTransferRecorder,
)


def test_stats_record_reduce_and_reset() -> None:
    stats = MooncakeKVConnectorStats(data={})
    stats.record_transfer(0.01, 2**20)
    stats.record_transfer(0.03, 3 * 2**20)
    stats.record_failed_transfer()

    reduced = stats.reduce()

    assert reduced["Num successful transfers"] == 2
    assert reduced["Num failed transfers"] == 1
    assert reduced["Avg xfer time (ms)"] == 20.0
    assert reduced["Avg MB per transfer"] == 2.0
    assert reduced["Throughput (MB/s)"] == 100.0

    previous = stats.clone_and_reset()
    assert previous.num_successful_transfers == 2
    assert stats.is_empty()


def test_stats_aggregate_ignores_empty_and_merges_nonempty() -> None:
    stats = MooncakeKVConnectorStats(data={})
    empty = MooncakeKVConnectorStats(data={})
    other = MooncakeKVConnectorStats(data={})
    other.record_transfer(0.1, 1024)
    other.record_failed_transfer()

    assert stats.aggregate(empty) is stats
    stats.aggregate(other)

    assert stats.data["transfer_duration"] == [0.1]
    assert stats.data["bytes_transferred"] == [1024]
    assert stats.data["num_failed_transfers"] == [1]


def test_transport_recorder_counts_error_codes_and_exceptions() -> None:
    recorder = MooncakeTransferRecorder()

    def raise_error(*args) -> int:
        raise RuntimeError("native transfer error")

    assert recorder.transfer(lambda *args: -1, "peer", [0], [1], [1024])[0] == -1
    with pytest.raises(RuntimeError, match="native transfer error"):
        recorder.transfer(raise_error, "peer", [0], [1], [1024])

    stats = recorder.drain()
    assert stats is not None
    assert stats.data["num_failed_transfers"] == [1, 1]
    assert not stats.data["bytes_transferred"]
    assert not stats.data["transfer_duration"]
    assert recorder.drain() is None
