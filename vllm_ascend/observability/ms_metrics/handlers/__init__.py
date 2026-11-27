# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM Ascend-owned metric handlers, one module per feature.

Shared building blocks live in :mod:`.common`; feature modules only depend on
them. Re-export new feature handlers here so the YAML reference
``vllm_ascend.observability.ms_metrics.handlers:function_name`` stays stable.
"""

from vllm_ascend.observability.ms_metrics.handlers.kv_transfer import (
    kv_transfer_finished_handler,
    kv_transfer_invalid_blocks_handler,
    kv_transfer_remote_waiting_handler,
    mooncake_receive_stats_handler,
    mooncake_waiting_handler,
)
from vllm_ascend.observability.ms_metrics.handlers.runtime_guard import (
    runtime_guard_handle_alert_handler,
)
from vllm_ascend.observability.ms_metrics.handlers.schedule_execute import sample_tokens_handler

__all__ = [
    "kv_transfer_finished_handler",
    "kv_transfer_invalid_blocks_handler",
    "kv_transfer_remote_waiting_handler",
    "mooncake_receive_stats_handler",
    "mooncake_waiting_handler",
    "runtime_guard_handle_alert_handler",
    "sample_tokens_handler",
]
