# PD 分离下 recompute scheduler 与 dummy graph 首请求异常进展

更新日期：2026-06-18

## 问题概览

当前在 `vllm-ascend` 基于 `971d50b3f98` 版本运行 PD 分离场景时，主要遇到两类问题：

1. **curl 请求不返回**

   该问题发生在 recompute scheduler 路径。当前已确认根因是该版本未同步 vLLM 主线修改，在 Mamba block aligned split 相关逻辑中缺少 `and not load_kv_async` 条件，导致 PD 分离 D 节点 load KV async 场景下仍进入不合适的切分逻辑，`num_new_tokens` 可能持续为 0，最终请求卡住不返回。

   该问题已通过同步主线判断条件解决。

2. **curl 能返回后，首请求输出异常**

   修复 scheduler 后，请求可以返回，但 D 节点开启图模式且 `dp > 1` 时仍存在首请求输出异常。观察到的现象包括：

   - 开启 MTP 时，首个请求输出大量异常字符，例如感叹号；后续请求基本正常。
   - 不开启 MTP 时，回复跳过正常 `think` 内容，直接从 `</thinking>` 开始。
   - 首次 KV cache 传输耗时约 `90ms`，后续请求约 `1.7ms`。
   - `dp=1` 时问题不复现；`dp>1` 时会走 dummy run，问题更容易复现。

## 当前定界

目前两个问题已经基本分开：

- **请求不返回问题**：定界在 recompute scheduler 未同步主线 `and not load_kv_async` 条件，已解决。
- **首请求输出异常问题**：倾向定界在 D 节点图模式、DP dummy run、Mamba/GDN state/KV cache 写入路径的组合问题。

支持第二个判断的现象包括：

1. `dp=1` 不走 dummy run，问题不复现。
2. `dp>1` 走 dummy run，首请求异常。
3. 开启 MTP 时问题只出现在第一次请求，后续请求正常。
4. 强制在 `block_pool` 开始处预留 32 个 block 后，问题消失。
5. 禁止 dummy run 走图模式后，问题也可以规避，进一步说明问题和 dummy graph/capture 期间的缓存或 state 写入有关。
6. vLLM BlockPool 中 block 0 是预留 `null_block`，真实 block 理论上从 1 开始分配。
7. 初步检查 GDN/Mamba/AscendC 相关算子后，没有发现 index 0 本身导致越界的明确证据。0 更像是合法地址，会写到第 0 个 block/state，而不是 no-op。

基于以上现象，当前更怀疑 dummy graph/capture 过程中 active row 的 `block_table` 未清理，复用了历史非 0 block id，导致 dummy run 写到了低位真实 block，例如 1 到 31。预留 32 个 block 后真实请求避开这些低位 block，所以问题被规避。

## 已尝试方案和结果

### 1. 同步 recompute scheduler 主线逻辑

在 Mamba block aligned split 判断中补齐主线条件 `and not load_kv_async`。

结果：

- curl 不返回问题已解决。

判断：

- 该问题根因相对明确，属于 vLLM Ascend 当前版本未同步 vLLM 主线修复。

### 2. 禁止 dummy run 走图模式

尝试将 dummy run 改为非图模式，例如让 dummy 走 `CUDAGraphMode.NONE`。

结果：

- 首请求输出异常可以规避。

判断：

- 该方案说明问题和 dummy graph/capture 路径强相关。
- 但该方案可能影响图模式 warmup/capture 覆盖完整性，带来性能或后续 replay 行为风险，因此更适合作为定位手段或临时 workaround，不建议作为最终修复。

### 3. dummy run 禁止 cache 写入

尝试在 dummy attention metadata 中将 `slot_mapping` 置为 `-1`，避免 dummy run 写 KV cache。

结果：

- 请求可以跑通，但仍需确认是否覆盖 GDN/Mamba state index 路径。

判断：

- GDN state index 主要来自 `block_table`，不完全依赖 `slot_mapping`。
- `-1` 对部分 AscendC 算子可能存在兼容性风险。
- 后续如果继续保留该逻辑，需要确认所有相关算子对 `-1` 的语义一致。

### 4. 预留低位 block

在 `block_pool` 初始化阶段强制预留前 32 个 block。

结果：

- 首请求输出异常消失。

判断：

- 这是有效 workaround，但不是正式修复。
- 该现象强烈暗示问题和 dummy run 写低位 block 有关。
- 该方案会浪费 KV cache 容量，也不能解释根因，不建议作为最终方案。

### 5. 检查 GDN/Mamba/AscendC 算子

初步检查已有相关 AscendC 算子后，没有发现 index 0 本身导致越界的明显代码路径。

当前判断：

- 0 本身不是越界，更像是“会真实写入 null block”。
- 真正危险的是 active dummy row 中出现非 0 的 stale block id，此时 dummy run 可能写到低位真实 block。

### 6. 新增诊断断言

目前已在 dummy run 构建 attention metadata 时新增检查，只在 dummy 路径打开。检查内容是：

```python
block_table_tensor[:num_reqs, 0]
```

如果 active dummy rows 中出现非 0 block id，会打印以下信息并触发断言：

- `kv_cache_gid`
- `num_reqs`
- `num_reqs_padded`
- `num_tokens`
- `num_tokens_padded`
- `bad_rows`
- `bad_values`
- `first_col_sample`

断言信息：

```text
AssertionError: dummy block_table active rows contain non-zero block ids
```

目标：

- 验证 dummy/capture 是否复用了旧 block_table。
- 如果断言命中，基本可以证明 dummy active rows 写到了非 null 的低位真实 block。

## 当前解决情况

目前状态如下：

- curl 不返回问题已解决，根因是 recompute scheduler 没有同步主线 `and not load_kv_async`。
- 首请求输出异常尚未最终定根因。
- 禁止 dummy 走图模式、预留 32 block 均可规避首请求输出异常，说明问题与 dummy graph/capture 以及低位 block/state 写入高度相关。
- 已加诊断用于确认 dummy run active block table 是否存在非 0 stale block id。
- 需要在复现场景中跑首请求，观察是否命中新增断言。

## 当前隐患

1. 禁止 dummy 图模式可能影响 graph warmup/capture 的覆盖完整性，可能带来性能回退或后续 replay 风险。
2. `slot_mapping = -1` 不一定覆盖 GDN/Mamba state index 路径。
3. `-1` 对部分 AscendC 算子可能不兼容。
4. 如果 dummy run active block_table 复用旧非 0 block id，则会污染真实 block。
5. 如果 dummy run 只写 block 0，理论上真实请求不应读取 block 0；但如果某些路径误读 null block，也可能导致输出异常。
6. 预留 32 block 是 workaround，会降低 KV cache 可用容量，不适合作为最终方案。
7. 当前问题只在首请求或特定图模式下复现，可能和 dummy graph capture、buffer 复用、PD 首次传输、DP padding 共同相关。

## 下一步思路

1. 使用新增诊断跑复现场景，确认 dummy run active `block_table_tensor[:num_reqs, 0]` 是否存在非 0。
2. 如果确认存在非 0，建议正式修复为：dummy/cudagraph capture 路径显式清理 active block table rows 到 `NULL_BLOCK_ID=0`，避免复用历史真实 block id。
3. 同时评估是否需要保留或调整 `slot_mapping = -1` 逻辑，避免 `-1` 对算子的兼容风险。
4. 如果 active block_table 已经全为 0，但问题仍复现，则继续排查真实路径是否读取了 block 0/null state，或 GDN/Mamba 某些算子在 index 0 时存在状态污染传播。
5. 继续检查 Qwen3.5 使用到的 GDN/Mamba/AscendC 算子边界语义，重点确认 index 0、`-1`、block offset 计算、dummy/real 是否共享 state/KV cache buffer。

## 恳请专家判断

当前希望专家协助判断以下问题：

1. dummy active block table 清零到 `NULL_BLOCK_ID=0` 是否符合 vLLM BlockPool 的设计语义。
2. 该修复应放在上游 dummy block table 创建处，还是先在 vLLM Ascend model runner 侧兜底。
3. 对于 GDN/Mamba state 写入路径，是否应该额外设计“dummy 专用 state/cache buffer”，而不是复用 null block 0。
4. `slot_mapping = -1` 是否适合作为 dummy cache write disable 的统一语义，还是需要针对不同 AscendC 算子分别处理。
5. 如果预留低位 block 能规避问题，是否可以进一步佐证“dummy run 写到了低位真实 block”的判断。
