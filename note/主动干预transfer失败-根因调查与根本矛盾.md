# 主动干预 transfer 失败：根因调查与根本矛盾

> 记录「主动 KV 干预」在真实 sglang + mooncake（fork B / Pro5000）上，
> P2P transfer 几乎全部失败（`source cache does not cover`）的完整调查过程、代码级根因、
> 以及最终暴露的「mamba 模型主动 transfer 机制性矛盾」。

---

## 1. 背景与当前任务

- **项目主线**：LLM 会话行为分析 → 会话级 KV Cache 调度 → 主动 KV 干预。
  最终目标三步：整理实验数据 → 跑模拟测试 → 建立离线收益/损失模型。
- **当前阶段**：在真实 sglang + mooncake（fork B，`--enable-prefill-p2p-kv-transfer`）上验证
  主动干预（proactive broadcast / recall）能否带来命中率提升。
- **基线**：纯 prefix cache + 被动 P2P（`--p2p-transfer-policy auto`）。

## 2. 数据集整理（已完成）

`experiments/offline_modeling_dataset.tar.gz`（708KB），含：

| 文件 | 行数 | 内容 |
|---|---|---|
| `decisions.jsonl` | 79312 | 决策日志（hit_count/depth/benefit/cost）|
| `routing.jsonl` | 1739 | 路由日志（route_reason/matched_tokens）|
| `transfer.jsonl` | 1508 | transfer 日志（ok/rejected/tokens/耗时）|
| `per_request.jsonl` | 548 | 每请求结果（548 条真实 agent 请求，含 tools）|

实验设置：v27（hash 修复后）、6P+1D、hicache-ratio 2.0、128K 上下文、156 用户交错 c=1。

## 3. transfer 失败现象

**v27 full 模式（depth-cap=1024）1508 次 transfer：**

| reason | ok | rejected | 成功率 |
|---|---|---|---|
| broadcast | 0 | 615 | **0.00%** |
| proactive | 14 | 879 | **1.57%** |

- 成功的 14 次全是 128/256 token 的浅前缀（偶然撞对时机），深前缀（448+）0 成功。
- 失败原因 100% 是 `source residency probe failed: HTTP 400: {...source cache does not cover the requested transferable prefix: device=0 host=0 ...}`。
- 命中率 54.92%（与不干预组相同，说明主动干预未产生任何增量收益，命中全来自被动 prefix cache）。

## 4. 根因调查（逐步排除）

### 4.1 先排除了 gateway 侧的问题（决定性验证）

用 v29 加了 `transfer token_ids` 诊断日志，抓出 gateway 实际发出的 token_ids：

```
token_ids_prefix = [248045, 8678, 198, 2, 13455, 271, 2523, 599, ...]
block_hashes_prefix = [-437949989974569719, -1899827859074762601, 1820820755417947648, ...]
```

在 worker 上用 transformers 复刻 sglang 渲染 + tokenize 第一个请求的模板，前 16 个 token **完全一致（match=True）**：
`248045=<|im_start|> 8678=system 198=\n 2=# ...`。

**结论：gateway 发的 token_ids 和 hash 都完全正确，排除 gateway 侧。**

### 4.2 定位到 fork-B 的 source 端 match 机制

关键代码（fork B `p2p_kv_transfer.py` + `hi_mamba_radix_cache.py`，都在 worker 容器
`/sgl-workspace/sglang/python/...`）：

1. **target 端 `_source_residency_probe()`**：target 收到 transfer 请求后，**不盲信 gateway 的命中信息**，
   而是发 dry_run probe 到 source，让 source **重新 `match_prefix` 验证**自己是否真有这个前缀。
   - 完整 GPU hit → `DEVICE` 模式；
   - 完整 host-only hit 且无 mamba → `HICACHE_DIRECT` 模式；
   - 前缀不足 → 返回 `source cache does not cover`（fallback，本地重算）。

2. **source 端 `_match_prefix_helper()`** 的匹配中断条件：
   ```python
   if child.evicted and not child.backuped:
       break
   ```
   其中 `evicted = value is None`（full-KV 被释放）、`backuped = host_value is not None`（host 有备份）。

3. **为什么 device=0 host=0**：source 的 cache 里，模板前缀已经被淘汰/删除（`_evict_regular`
   对「未 backuped」的 node 直接 free GPU + 从 tree 删除），`match_prefix` 匹配不到。

### 4.3 关键：KV events 是周期性批量发布的，不是实时

`_publish_kv_events()`（`scheduler_metrics_mixin.py:781`）在 **metrics 上报周期**里调用：

```python
events = self.tree_cache.take_events()   # 把累积的 BlockStored/BlockRemoved 一次性取走
self.kv_event_publisher.publish(batch)
```

`BlockRemoved` 从 `_evict_regular` 产生 → 进 `kv_event_queue` → **等下一个 metrics 周期才发布** →
经 ZMQ → gateway pump → tree 更新。gateway 的决策（5s tick）与 source 的淘汰之间存在**竞态窗口**。

### 4.4 为什么被动 P2P 不受影响

被动 P2P（PD 分离，`disaggregation/prefill.py:send_kv_chunk`）传输的是「当前请求刚 prefill 完、
**此刻必然还在 device 的 KV**」，用 `req_to_token_pool.req_to_token[req.req_pool_idx]` 直接拿物理指针，
**不查历史 cache、不等事件、不走 match_prefix**：

```python
kv_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx, ...]
req.disagg_kv_sender.send(page_indices, state_indices)
```

而主动干预（broadcast/recall/proactive）走 `/experimental/p2p_kv_transfer` 端点，**用 token_ids 倒查历史 cache**，
必然撞上「事件索引滞后 + source 已淘汰」的竞态。

> 参考：同事 fork A 的文档 `p2pkvcache.md` §5.4/§5.5/§7.1 明确写了同样的设计——
> source 无完整 prefix 是「设计内的 fallback（本地重算，不影响推理）」，不是 bug。
> 所以「98.4% transfer 被拒」本质是「gateway 决策质量差（发了太多注定 fallback 的 transfer）」，
> 不是机制坏了。

## 5. 已做的改动（三批）

### A. 自愈反馈（✅ 已验证生效）

transfer 被拒且 message 含 `source cache does not cover` 时，把 source 从 tree 摘除，避免重复选它。

- `tree.rs`：新增 `HashTree::remove_worker_by_url(url, block_hashes)`。
- `proactive.rs`：`WarmRequest`/`Placement` 加 `block_hashes`，warmer 持有 tree 句柄，被拒时自愈。
- 日志：`dropped stale source` 紧跟每次 rejected。

### C. 浅前缀广播（部分有效）

`gen_gateway.py`/`orchestrate.sh` 的 `--depth-cap` 默认 4096 → 128。效果：transfer 次数 1508 → 6
（broadcast 一次铺满失败后 source 被自愈摘除、decider 无 source 可选）。但浅前缀 transfer 也全失败，
**证明失败与深度无关**。

### 方向 B. 年龄门（✅ 验证生效，但未解决 transfer）

「只广播能长期存在的链」：tree 的 node 加 `created_ms`，决策器加 `min_live_ms` 阈值。

- `tree.rs`：`Node.created_ms`、`HotChain.created_ms`、`now_millis` 改 `pub(crate)`。
- `proactive.rs`：`evaluate(..., now_ms)` 加年龄过滤（`age < min_live_ms → too_young`），
  broadcast 控制器同样加年龄门。
- `config/types.rs`/`cli.rs`/`gen_gateway.py`：新增 `--proactive-min-live-ms`（默认 30000）。
- 测试：`evaluate_skips_chains_younger_than_min_live_ms`（447 单测全过）。

验证结果：前几个 tick `too_young=2`，30 秒后 `placements=2`，broadcast 确实延迟到模板成熟后才触发。
**但 transfer 仍然 100% 失败（device=0 host=0）。**

## 6. 根本矛盾：mamba 模型的主动 transfer 机制性不可行

查 source（p-worker-4）cache 指标（v30 验证时）：

| 指标 | 值 |
|---|---|
| device full-KV 池使用 | **29170 token**（≈ 一个请求的量）|
| host 池使用 | **905372 token** |

矛盾链条：

1. **`--hicache-write-policy write_through` 把 full-KV 主动 demote 到 host**：
   source 的 device 里只留「当前正在 prefill 的请求」的 full-KV（29K），
   历史 cache（包括模板）的 full-KV 全在 host（905K）。

2. **transfer 只有两条路，对 mamba 模型全堵死**：
   - `DEVICE` 模式需要「device 里的完整 full-KV」→ 历史 cache 的 full-KV 已 demote → `device_tokens=0`；
   - `HICACHE_DIRECT` 模式（host 直传）→ `_source_residency_probe` 里
     `policy == "hicache" and not _requires_mamba()`，而 qwen3.5-4b 是 hybrid SSM（32 层 = 24 linear_attention + 8 full_attention），
     `requires_mamba=True` → **host 直传被拒**。
     （fork A 文档 §8.2 也写明「HICACHE_DIRECT 不支持 Mamba state」。）

3. **结论**：「长期存在」≠「device 里还有 full-KV」。对 mamba 模型，历史 cache 的 full-KV 永远不在 device，
   而 host 直传又不支持 mamba —— 这是**数据在哪个介质的问题，不是广播时机的问题**。
   因此「等 30 秒」「只广播长期链」都救不了 transfer。

## 7. 下一步选项

1. **请 fork-B 同事开 `HICACHE_DIRECT` 支持 mamba state**：host 里有 905K token 的 full-KV + mamba host，
   理论上能传，只是当前 probe 用 `not requires_mamba` 挡死了。这是唯一能真正解 transfer 的路，但靠同事。
2. **接受「mamba 下主动干预无收益」**：用现有数据把离线模型收敛成「收益 ≈ 0」的结论，
   回到「整理数据 → 模拟 → 离线模型」主线。
3. **先 commit 已做的保守化改动**（A 自愈 + C 浅前缀 + 方向 B 年龄门），它们本身是正确的防御性改进。

## 8. 关键代码位置索引

| 文件 | 关键点 |
|---|---|
| `sgl-router-src/src/policies/kv_events/tree.rs` | `Node.created_ms`、`remove_worker_by_url`、`list_hot_chains` |
| `sgl-router-src/src/policies/proactive.rs` | `evaluate` 年龄门、自愈摘除、`too_young` 统计 |
| `sgl-router-src/src/config/types.rs` / `cli.rs` | `min_live_ms` 配置 |
| `experiments/gen_gateway.py` | `--depth-cap` / `--min-live-ms` 默认值 |
| worker `/sgl-workspace/sglang/python/sglang/srt/disaggregation/p2p_kv_transfer.py` | `_source_residency_probe`（probe 裁决）、`_transferable_len` |
| worker `.../mem_cache/hi_mamba_radix_cache.py` | `_match_prefix_helper`（break 条件）、`_evict_regular`/`_evict_to_host`、`_inc_hit_count`（write_through 阈值）|
| worker `.../mem_cache/radix_cache.py` | `evicted`/`backuped` 属性、`RadixKey.match`/`child_key` |
| worker `.../disaggregation/prefill.py` | `send_kv_chunk`（被动 P2P，不查历史 cache）|
| worker `.../observability/scheduler_metrics_mixin.py` | `_publish_kv_events`（周期批量发布）|
| `p2pkvcache.md`（fork A 文档）| §5.4/5.5/7.1：probe 是设计内 fallback；§8.2：HICACHE_DIRECT 不支持 mamba |
