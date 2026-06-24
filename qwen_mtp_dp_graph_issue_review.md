# Qwen MTP DP Graph Issue Review

## 背景

在 Qwen3.5 / GDN 场景中，开启 MTP、DP > 1、ACL Graph 后，首批请求出现精度异常，典型表现为输出持续变成 `!!!!!`。关闭 MTP 或开启 `enforce_eager` 后问题消失，说明问题集中在 MTP + 图模式路径。

此前另有一个 curl 不返回问题，根因是 recompute scheduler 未同步主线逻辑，`need_mamba_block_aligned_split` 分支缺少 `and not load_kv_async` 保护，导致 PD 分离场景下 `num_new_tokens` 可能持续为 0。该问题已单独修复，和本次 MTP 精度问题不是同一个根因。

## 定位过程

1. 首先怀疑 dummy run / graph capture 影响 GDN/Mamba state。
   - DP=1 正常，DP>1 异常。
   - 预留低位 block 后问题缓解，说明存在低位 state/block 被错误写入的可能。
   - 非 MTP 场景中，在 dummy run 的 attention metadata 构建前同步 block table 后，首请求问题可恢复。

2. 针对 MTP 场景继续排查后发现，非 MTP 修复不能解决 `!!!!!`。
   - 加入诊断后发现 `update_conv1d_graph_params` 中出现：
     - capture 分支为 `branch == "spec"`；
     - runtime `meta.spec_sequence_masks is None`。
   - 这说明图里捕获了 spec conv1d task，但当前 DP rank / runtime batch 没有真实 spec sequence。

3. 在算子前后打印 GDN state，确认异常不是纯 metadata 问题，而是 state 被写坏。
   - 在 `npu_causal_conv1d_custom` 调用前后，对 `conv_state` 做快照和摘要打印，例如检查低位 block 的 sum / min / max / 是否变化。
   - 异常场景中，即使当前 rank 没有真实 spec sequence，spec conv1d graph task replay 后，`conv_state` 仍发生变化。
   - 这说明问题不是“没有输出”或“采样阶段错了”，而是 dummy/空 spec task 真实修改了 persistent GDN conv state。
   - 结合“预留低位 block 后问题消失”的实验，可以进一步说明错误写集中在低位或错误 state block，真实请求后续复用这些 state 后出现 `!!!!!`。

4. 继续追 `spec_sequence_masks` 来源。
   - `spec_sequence_masks` 由 GDN metadata builder 根据 `num_decode_draft_tokens_cpu` 生成。
   - 当当前 batch/rank 没有有效 draft tokens 时，`num_decode_draft_tokens_cpu` 全为 `-1`，builder 会设置 `spec_sequence_masks = None`。
   - 因此 `None` 本身是合法语义：表示当前 runtime 没有 spec decode work。

5. 结合算子实现确认真正问题点。
   - `csrc/moe/causal_conv1d` 中，`cache_indices` 为空时不是 no-op。
   - 空 `cache_indices` 会走默认 batch-indexed decode 写回，可能写 `conv_state[0..x.shape[0])`。
   - 对于 `branch == "spec"` 但 runtime 没有 spec sequence 的场景，dummy/空 rank 不应该修改 persistent `conv_state`。

6. 最后和算子同学确认 no-op 参数语义。
   - 对于不需要使用 dummy/spec task 输出的场景，只要 `cache_indices` 按 `mixed_qkv.shape[0]` 全填 `-1` 即可。
   - 算子侧会把 `cache_indices[i] == PAD_SLOT_ID` 的 row 跳过，因此不会读写对应 conv state。
   - 这也解释了为什么不能传空 `cache_indices`：空表示“没有显式 indices”，不是“全部跳过”。

## 根因

ACL Graph capture 阶段可能捕获到 GDN spec conv1d graph task；但在 DP + MTP runtime 中，某些 rank / step 没有真实 spec sequence，`meta.spec_sequence_masks` 会合法地为 `None`。

原逻辑在该状态下没有进入 spec 参数更新分支，也没有进入 non-spec decode 分支，导致传给 `npu_causal_conv1d_custom` 的 `cache_indices` 为空。算子侧将空 `cache_indices` 解释为默认 batch-indexed state write，而不是 no-op，最终可能写坏低位或错误的 GDN `conv_state`，首批请求读取到被污染的 state 后输出坍缩成 `!!!!!`。

从代码和实验结果看，问题链路是：

```text
MTP + DP + ACL Graph
        ↓
capture 阶段捕获 spec conv1d graph task
        ↓
某些 runtime rank / step 没有真实 spec sequence
        ↓
meta.spec_sequence_masks = None
        ↓
update_conv1d_graph_params 未生成 spec cache_indices
        ↓
算子收到空 cache_indices
        ↓
算子按默认 batch index 写 conv_state
        ↓
低位或错误 GDN state 被污染
        ↓
真实请求读取污染后的 state，输出变成 !!!!!
```

## 修复方案

修复点放在 `vllm_ascend/ops/gdn.py::update_conv1d_graph_params`。

当满足：

```text
branch == "spec"
meta.spec_sequence_masks is None
```

时，说明当前 captured spec conv1d task 没有真实 runtime spec work。此时不改 metadata 语义，也不强行构造 spec mask，而是在 graph task update 层显式传入：

```python
new_cache_indices = (PAD_SLOT_ID,) * cap_x_dim0
```

其中 `cap_x_dim0 = mixed_qkv.shape[0]`，`PAD_SLOT_ID = -1`。算子看到每个 row 的 cache index 都是 padding 后，会跳过对应 state 写回，使该 captured spec task 成为 `conv_state` no-op。

## 影响范围

- 只影响 GDN conv1d ACL Graph replay/update 路径。
- 只在 captured branch 为 spec、但 runtime 没有 spec sequence 时生效。
- 不影响正常 spec decode，因为 `meta.spec_sequence_masks is not None` 时仍走原有参数更新逻辑。
- 不影响 non-spec decode，也不影响 eager 路径。

## 经验总结

- `spec_sequence_masks is None` 不一定是异常，必须结合 capture branch 判断语义。
- graph task update 不能把空 host args 当成 no-op，必须确认算子对空 optional input 的真实行为。
- 对 persistent state 类算子，dummy/capture/replay 阶段即使没有真实请求，也可能破坏后续请求状态。
- 修复应尽量贴近根因：metadata 表达 runtime 事实，graph update 层负责把无真实 work 的 captured task 安全 no-op。

## 当前修复提交

```text
5a5661a809d4fcaf6d106b10e8221b5b3f5d5519
Fix spec conv1d cache indices for empty spec batches
Signed-off-by: zhuyixiang <zhuyixiang2014@163.com>
```
