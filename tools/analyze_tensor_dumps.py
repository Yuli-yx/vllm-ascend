import argparse
import os
from typing import Any

import torch


def walk(prefix: str, obj: Any):
    if isinstance(obj, dict):
        if {"shape", "dtype", "numel", "nan", "inf", "zero_count", "absmax", "mean"} <= set(obj):
            yield prefix, obj
            return
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from walk(name, value)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from walk(f"{prefix}[{idx}]", value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir", nargs="?", default="/tmp/vllm_ascend_tensor_dump")
    parser.add_argument("--show-ok", action="store_true")
    args = parser.parse_args()

    paths = []
    for name in os.listdir(args.dump_dir):
        if name.endswith(".pt"):
            paths.append(os.path.join(args.dump_dir, name))
    paths.sort(key=lambda p: os.path.getmtime(p))

    first_bad = None
    for path in paths:
        obj = torch.load(path, map_location="cpu")
        rel = os.path.basename(path)
        for name, meta in walk("", obj):
            numel = int(meta["numel"])
            zero = int(meta["zero_count"])
            bad = bool(meta["nan"]) or bool(meta["inf"]) or (numel > 0 and zero == numel)
            if bad or args.show_ok:
                status = "BAD" if bad else "OK "
                print(
                    f"{status} {rel} :: {name} shape={meta['shape']} dtype={meta['dtype']} "
                    f"nan={meta['nan']} inf={meta['inf']} zero={zero}/{numel} "
                    f"absmax={meta['absmax']} mean={meta['mean']}"
                )
            if bad and first_bad is None:
                first_bad = (rel, name, meta)

    if first_bad is None:
        print("No NaN/Inf/all-zero tensor summaries found.")
    else:
        rel, name, meta = first_bad
        print("\nFIRST_BAD")
        print(
            f"{rel} :: {name} shape={meta['shape']} dtype={meta['dtype']} "
            f"nan={meta['nan']} inf={meta['inf']} zero={meta['zero_count']}/{meta['numel']} "
            f"absmax={meta['absmax']} mean={meta['mean']}"
        )


if __name__ == "__main__":
    main()
