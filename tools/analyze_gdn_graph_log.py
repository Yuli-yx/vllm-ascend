#!/usr/bin/env python3
"""Summarize vLLM Ascend GDN ACL graph debug logs.

This script only reads local log files and prints a compact summary. It does
not upload or persist log contents.
"""

from __future__ import annotations

import argparse
import ast
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TAG_RE = re.compile(r"\[(GDN_(?:GRAPH_(?:UPDATE_CALL|CAPTURE|UPDATE)|RECURRENT_STATE|METADATA_STATE))\]")
KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,]+?)(?=, [A-Za-z_][A-Za-z0-9_]*=|$)")


def parse_tuple(value: str) -> tuple[int, ...] | tuple:
    value = value.strip()
    if not value.startswith("("):
        return ()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return ()
    if isinstance(parsed, tuple):
        return parsed
    return ()


def parse_bool(value: str) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def parse_line(line: str) -> dict[str, object] | None:
    tag_match = TAG_RE.search(line)
    if tag_match is None:
        return None

    event: dict[str, object] = {
        "tag": tag_match.group(1),
        "line": line.rstrip("\n"),
    }
    tail = line[tag_match.end():].strip()
    for key, raw_value in KV_RE.findall(tail):
        value = raw_value.strip()
        if value.startswith("("):
            event[key] = parse_tuple(value)
        elif value in ("True", "False"):
            event[key] = parse_bool(value)
        else:
            try:
                event[key] = int(value)
            except ValueError:
                event[key] = value
    return event


def read_events(paths: Iterable[Path]) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in paths:
        with path.open("r", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                event = parse_line(line)
                if event is None:
                    continue
                event["file"] = str(path)
                event["line_no"] = line_no
                events.append(event)
    return events


def tuple_sample(value: object, limit: int = 12) -> str:
    if not isinstance(value, tuple):
        return "()"
    sample = value[:limit]
    suffix = "..." if len(value) > limit else ""
    return f"{sample}{suffix} len={len(value)}"


def count_nonzero(value: object) -> int:
    if not isinstance(value, tuple):
        return 0
    return sum(1 for x in value if isinstance(x, int) and x > 0)


def extract_sample(line: object, field: str) -> list[int]:
    if not isinstance(line, str):
        return []
    match = re.search(rf"{re.escape(field)}=.*?sample=(\[[^\]]*\])", line)
    if match is None:
        return []
    try:
        parsed = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [x for x in parsed if isinstance(x, int)]


def summarize_index_samples(events: list[dict[str, object]], title: str, fields: tuple[str, ...]) -> None:
    if not events:
        return
    print(f"\n== {title} index samples ==")
    for field in fields:
        samples = [extract_sample(e.get("line"), field) for e in events]
        samples = [sample for sample in samples if sample]
        if not samples:
            continue
        rows_with_zero = sum(0 in sample for sample in samples)
        rows_with_negative = sum(any(x < 0 for x in sample) for sample in samples)
        rows_with_positive = sum(any(x > 0 for x in sample) for sample in samples)
        print(
            f"{field}: rows={len(samples)} rows_with_zero={rows_with_zero} "
            f"rows_with_negative={rows_with_negative} rows_with_positive={rows_with_positive}"
        )
        print(f"  first_sample={samples[0]}")


def summarize(events: list[dict[str, object]]) -> None:
    print("== GDN graph log summary ==")
    print(f"total_events: {len(events)}")
    if not events:
        print("conclusion: no GDN_GRAPH_* logs found")
        return

    tag_counts = Counter(str(e["tag"]) for e in events)
    print(f"tag_counts: {dict(tag_counts)}")

    update_calls = [e for e in events if e["tag"] == "GDN_GRAPH_UPDATE_CALL"]
    captures = [e for e in events if e["tag"] == "GDN_GRAPH_CAPTURE"]
    updates = [e for e in events if e["tag"] == "GDN_GRAPH_UPDATE"]
    recurrent_states = [e for e in events if e["tag"] == "GDN_RECURRENT_STATE"]
    metadata_states = [e for e in events if e["tag"] == "GDN_METADATA_STATE"]

    if update_calls:
        print("\n== update call conditions ==")
        cond_counts = Counter(
            (
                e.get("use_compress"),
                e.get("has_gdn"),
                e.get("use_sparse"),
                e.get("enable_enpu"),
            )
            for e in update_calls
        )
        for cond, count in cond_counts.most_common():
            print(
                "count=%s use_compress=%s has_gdn=%s use_sparse=%s enable_enpu=%s"
                % (count, cond[0], cond[1], cond[2], cond[3])
            )
    else:
        print("\nwarning: no GDN_GRAPH_UPDATE_CALL lines; full graph runtime hook may not run")

    if captures:
        print("\n== capture cache indices ==")
        by_branch = defaultdict(list)
        for e in captures:
            by_branch[(e.get("branch"), e.get("num_actual_tokens"))].append(e)
        for key, rows in sorted(by_branch.items(), key=lambda item: str(item[0])):
            nonzero_rows = sum(count_nonzero(e.get("cidx")) > 0 for e in rows)
            print(
                f"branch={key[0]} num_actual_tokens={key[1]} count={len(rows)} "
                f"nonzero_cidx_rows={nonzero_rows}"
            )
            first = rows[0]
            print(
                "  sample layer=%s qsl=%s cidx=%s"
                % (
                    first.get("layer"),
                    tuple_sample(first.get("qsl")),
                    tuple_sample(first.get("cidx")),
                )
            )
    else:
        print("\nwarning: no GDN_GRAPH_CAPTURE lines; GDN conv1d graph capture may not happen")

    skip_updates = [
        e for e in updates if isinstance(e.get("line"), str) and "skip:" in str(e.get("line"))
    ]
    start_updates = [
        e for e in updates if isinstance(e.get("line"), str) and "start:" in str(e.get("line"))
    ]
    captured_updates = [
        e for e in updates if isinstance(e.get("line"), str) and "captured:" in str(e.get("line"))
    ]
    runtime_updates = [
        e for e in updates if isinstance(e.get("line"), str) and "runtime:" in str(e.get("line"))
    ]
    no_runtime_arg_updates = [
        e for e in updates if isinstance(e.get("line"), str) and "no runtime args generated" in str(e.get("line"))
    ]

    print("\n== update status ==")
    print(f"start_lines: {len(start_updates)}")
    print(f"skip_lines: {len(skip_updates)}")
    print(f"captured_param_lines: {len(captured_updates)}")
    print(f"runtime_param_lines: {len(runtime_updates)}")
    print(f"no_runtime_arg_lines: {len(no_runtime_arg_updates)}")
    if skip_updates:
        print("skip_samples:")
        for e in skip_updates[:8]:
            print(f"  {e.get('line')}")

    if runtime_updates:
        print("\n== runtime cache indices ==")
        by_branch = defaultdict(list)
        for e in runtime_updates:
            by_branch[(e.get("branch"), e.get("layer"))].append(e)
        for key, rows in list(by_branch.items())[:20]:
            empty_rows = sum(not bool(e.get("new_cidx")) for e in rows)
            nonzero_rows = sum(count_nonzero(e.get("new_cidx")) > 0 for e in rows)
            print(
                f"branch={key[0]} layer={key[1]} count={len(rows)} "
                f"empty_cidx_rows={empty_rows} nonzero_cidx_rows={nonzero_rows}"
            )
            first = rows[0]
            print(
                "  sample new_qsl=%s new_cidx=%s"
                % (
                    tuple_sample(first.get("new_qsl")),
                    tuple_sample(first.get("new_cidx")),
                )
            )
    else:
        print("\nwarning: no runtime update rows; graph params are not being refreshed")

    if metadata_states:
        print("\n== metadata state events ==")
        kind_counts = Counter()
        for e in metadata_states:
            line = str(e.get("line"))
            if "fallback=spec" in line:
                kind_counts["fallback=spec"] += 1
            elif "fallback=non_spec_decode" in line:
                kind_counts["fallback=non_spec_decode"] += 1
            elif "fallback=non_spec_prefill" in line:
                kind_counts["fallback=non_spec_prefill"] += 1
            elif "fill_unused_non_spec" in line:
                kind_counts["fill_unused_non_spec"] += 1
            elif " build," in line:
                kind_counts["build"] += 1
            else:
                kind_counts["other"] += 1
        print(f"kind_counts: {dict(kind_counts)}")
        for e in metadata_states[:8]:
            print(f"  sample: {e.get('line')}")
        summarize_index_samples(
            metadata_states,
            "metadata",
            (
                "non_spec_state_indices",
                "spec_state_indices",
                "cache_indices_cpu",
                "num_accepted_tokens",
                "num_accepted_tokens_cpu",
            ),
        )
    else:
        print("\nwarning: no GDN_METADATA_STATE lines; metadata builder diagnostics are absent")

    if recurrent_states:
        print("\n== recurrent state events ==")
        by_branch = Counter(e.get("branch") for e in recurrent_states)
        print(f"branch_counts: {dict(by_branch)}")
        by_capture = Counter((e.get("branch"), e.get("capturing"), e.get("is_draft_model")) for e in recurrent_states)
        for key, count in by_capture.most_common():
            print(f"count={count} branch={key[0]} capturing={key[1]} is_draft_model={key[2]}")
        for e in recurrent_states[:8]:
            print(f"  sample: {e.get('line')}")
        summarize_index_samples(
            recurrent_states,
            "recurrent",
            (
                "spec_state_indices",
                "non_spec_state_indices",
                "num_accepted_tokens",
                "spec_token_indx",
                "non_spec_token_indx",
            ),
        )
    else:
        print("\nwarning: no GDN_RECURRENT_STATE lines; recurrent op diagnostics are absent")

    print("\n== quick diagnosis ==")
    if not update_calls:
        print("- update hook did not run; check cudagraph mode and model_runner path")
    elif skip_updates:
        print("- update hook ran but skipped; inspect skip_samples above")
    elif runtime_updates and all(not bool(e.get("new_cidx")) for e in runtime_updates):
        print("- update ran but produced empty runtime cache indices")
    elif runtime_updates:
        print(
            "- update ran and produced runtime cache indices; if output is "
            "still bad, suspect graph_task_update effectiveness or another state path"
        )
    else:
        print("- insufficient update details; inspect raw GDN_GRAPH_UPDATE lines on server")
    if recurrent_states:
        spec_zero_rows = sum(0 in extract_sample(e.get("line"), "spec_state_indices") for e in recurrent_states)
        non_spec_zero_rows = sum(0 in extract_sample(e.get("line"), "non_spec_state_indices") for e in recurrent_states)
        negative_rows = sum(
            any(x < 0 for x in extract_sample(e.get("line"), "spec_state_indices"))
            or any(x < 0 for x in extract_sample(e.get("line"), "non_spec_state_indices"))
            for e in recurrent_states
        )
        if spec_zero_rows or non_spec_zero_rows:
            print(
                "- recurrent state indices include block 0 in samples "
                f"(spec_rows={spec_zero_rows}, non_spec_rows={non_spec_zero_rows}); "
                "compare first bad requests with later good requests"
            )
        if negative_rows:
            print(
                "- recurrent state indices include negative padding values in "
                "samples; check whether the target op supports padding"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="server log file(s)")
    args = parser.parse_args()

    events = read_events(args.logs)
    summarize(events)


if __name__ == "__main__":
    main()
