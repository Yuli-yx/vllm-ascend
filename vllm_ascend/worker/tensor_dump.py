import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import logger

from vllm_ascend import envs


_LAYER_RE = re.compile(r"(^|\.)(layers|h|blocks)\.\d+$")


class TensorDumper:

    def __init__(self, rank: int = 0) -> None:
        self.enabled = envs.VLLM_ASCEND_TENSOR_DUMP
        self.rank = rank
        self.step = 0
        self.max_steps = envs.VLLM_ASCEND_TENSOR_DUMP_STEPS
        self.max_elems = envs.VLLM_ASCEND_TENSOR_DUMP_MAX_ELEMS
        self.dump_dir = envs.VLLM_ASCEND_TENSOR_DUMP_DIR
        self._hooks: list[Any] = []
        self._torchair_create = None
        if self.enabled:
            os.makedirs(self.dump_dir, exist_ok=True)
            self._try_load_torchair()
            logger.warning("TensorDumper enabled: dir=%s rank=%s max_steps=%s max_elems=%s",
                           self.dump_dir, self.rank, self.max_steps, self.max_elems)

    def _try_load_torchair(self) -> None:
        try:
            import torchair  # type: ignore

            self._torchair_create = torchair.llm_datadist.create_npu_tensors
            logger.warning("TensorDumper will use torchair.llm_datadist.create_npu_tensors")
        except Exception as e:
            self._torchair_create = None
            logger.warning("TensorDumper torchair llm_datadist unavailable, fallback to direct tensor copy: %s", e)

    def active(self) -> bool:
        return self.enabled and self.step < self.max_steps

    def next_step(self) -> None:
        if self.enabled:
            self.step += 1

    def register_model_hooks(self, model: nn.Module) -> None:
        if not self.enabled:
            return
        for name, module in model.named_modules():
            if not self._should_hook(name):
                continue
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))
        logger.warning("TensorDumper registered %s forward hooks", len(self._hooks))

    def _should_hook(self, name: str) -> bool:
        if not name:
            return False
        low = name.lower()
        return bool(_LAYER_RE.search(name) or "mamba" in low or "linear_attn" in low or "self_attn" in low)

    def _make_hook(self, name: str):

        def hook(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.active():
                return
            safe_name = name.replace(".", "_")
            self.dump(f"layer_{safe_name}", {"input": inputs, "output": output})

        return hook

    def dump(self, tag: str, obj: Any) -> None:
        if not self.active():
            return
        try:
            payload = self._pack(obj)
            path = os.path.join(self.dump_dir, f"rank{self.rank:02d}_step{self.step:04d}_{tag}_{time.time_ns()}.pt")
            torch.save(payload, path)
            logger.warning("TensorDumper saved %s", path)
        except Exception:
            logger.exception("TensorDumper failed to dump tag=%s", tag)

    def dump_mamba_cache(self, tag: str, kv_cache_config: Any, forward_context: Mapping[str, Any]) -> None:
        if not self.active() or kv_cache_config is None:
            return
        items: dict[str, Any] = {}
        try:
            from vllm.v1.kv_cache_interface import MambaSpec

            for group_idx, group in enumerate(kv_cache_config.kv_cache_groups):
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                for layer_name in group.layer_names:
                    layer = forward_context.get(layer_name)
                    kv_cache = getattr(layer, "kv_cache", None)
                    if not kv_cache:
                        continue
                    for state_idx, state in enumerate(kv_cache):
                        if torch.is_tensor(state) and state.shape[0] > 0:
                            items[f"group{group_idx}.{layer_name}.state{state_idx}.block0"] = state[0]
                            if state.shape[0] > 1:
                                items[f"group{group_idx}.{layer_name}.state{state_idx}.block1"] = state[1]
            if items:
                self.dump(f"mamba_cache_{tag}", items)
        except Exception:
            logger.exception("TensorDumper failed to collect mamba cache tag=%s", tag)

    def _pack(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return self._pack_tensor(obj)
        if isinstance(obj, Mapping):
            return {str(k): self._pack(v) for k, v in obj.items()}
        if isinstance(obj, tuple):
            return tuple(self._pack(v) for v in obj)
        if isinstance(obj, list):
            return [self._pack(v) for v in obj]
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [self._pack(v) for v in obj]
        return obj

    def _pack_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        wrapped_error = None
        src = tensor.detach()
        if self._torchair_create is not None and src.device.type != "cpu":
            try:
                src = self._torchair_create(list(src.shape), src.dtype, [src.data_ptr()])[0]
            except Exception as e:
                wrapped_error = repr(e)
                src = tensor.detach()

        data = src.detach()
        if not data.is_contiguous():
            data = data.contiguous()

        cpu = data.cpu()
        flat = cpu.reshape(-1)
        finite = cpu
        if cpu.is_floating_point() or cpu.is_complex():
            nan = torch.isnan(cpu).any().item()
            inf = torch.isinf(cpu).any().item()
            zero = (cpu == 0).sum().item()
            absmax = finite.float().abs().max().item() if flat.numel() else 0
            mean = finite.float().mean().item() if flat.numel() else 0
        else:
            nan = False
            inf = False
            zero = (cpu == 0).sum().item() if flat.numel() else 0
            absmax = cpu.float().abs().max().item() if flat.numel() else 0
            mean = cpu.float().mean().item() if flat.numel() else 0

        payload: dict[str, Any] = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "stride": tuple(tensor.stride()),
            "data_ptr": int(tensor.data_ptr()) if tensor.device.type != "cpu" else 0,
            "wrapped_by_torchair_error": wrapped_error,
            "numel": flat.numel(),
            "nan": bool(nan),
            "inf": bool(inf),
            "zero_count": int(zero),
            "absmax": absmax,
            "mean": mean,
            "head": flat[: min(64, flat.numel())].clone(),
            "tail": flat[-min(64, flat.numel()):].clone() if flat.numel() else flat.clone(),
        }
        if flat.numel() <= self.max_elems:
            payload["tensor"] = cpu.clone()
        return payload
