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


TAG_RE = re.compile(r"\[(GDN_GRAPH_(?:UPDATE_CALL|CAPTURE|UPDATE))\]")
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

    print("\n== quick diagnosis ==")
    if not update_calls:
        print("- update hook did not run; check cudagraph mode and model_runner path")
    elif skip_updates:
        print("- update hook ran but skipped; inspect skip_samples above")
    elif runtime_updates and all(not bool(e.get("new_cidx")) for e in runtime_updates):
        print("- update ran but produced empty runtime cache indices")
    elif runtime_updates:
        print("- update ran and produced runtime cache indices; if output is still bad, suspect graph_task_update effectiveness or another state path")
    else:
        print("- insufficient update details; inspect raw GDN_GRAPH_UPDATE lines on server")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path, help="server log file(s)")
    args = parser.parse_args()

    events = read_events(args.logs)
    summarize(events)


if __name__ == "__main__":
    main()
