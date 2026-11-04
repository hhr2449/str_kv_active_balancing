# 主动 KV 干预 — 离线建模数据集（哈希链版，不含原文）

生成：2026-10-22 收集，2026-10-26 补入哈希链 trace。hash 修复（v27）后的干净收集。
脱敏：worker URL → p0..p5/d0；sub_uin → 哈希假名；原文 messages 不外发，仅保留哈希链（块=64）。

## 文件

| 文件 | 内容 |
|---|---|
| `trace.jsonl` | 548 条请求的哈希链 trace：timestamp / input_length / output_length / hash_ids / cached_target |
| `request_meta.json` | idx / sub_uin(脱敏) / start_time / n_tokens / cached_target |
| `decisions.jsonl` | 决策日志（hit_count/depth/benefit/cost/source/target/executed）|
| `routing.jsonl` | 路由日志（reason/matched_tokens_est/hit_count/selected）|
| `transfer.jsonl` | transfer 日志（ok/rejected、transferred_tokens/elapsed_ms/bandwidth、fail_message）|
| `per_request.jsonl` | 逐请求命中（prompt_tokens/cached_tokens/ttfb/ttlb/error）|

## 哈希链口径

- `hash_ids` 为块=64（mamba chunk）的链式 SHA256，逐块取前 8 字节大端转 signed i64。
  块大小 64 = qwen3.5-4b 的 `mamba_cache_chunk_size`，与 transfer 的 depth 对齐（MAMBA_CHUNK=64）。
- `hash_ids` 长度 = ceil(input_length / 64)；`cached_target` = glm5 大上下文目标命中标注（token 级）。
- 哈希算法与 gateway `sgl-router-src/src/policies/kv_events/hash.rs` 一致，链式：每个块 feed 前一块的完整 32 字节 digest + 本块 64 个 token 的 4 字节 little-endian。

## 量级

- 请求 548 条 / 唯一 sub_uin 1 个（单用户多轮 agent 会话）/ system prompt 变体 156 种
- 时间跨度单日（2026-05-13）；n_tokens 总和 2619 万，目标命中率 cached_target/n_tokens = 86.2%
- output_length 全 0（源数据集只标注输入，未标注输出长度）

## 实验设置

- 数据：glm5-workbuddy 真实 agent 请求（含 tools），156 模板交错回放，c=1
- 模型 qwen35-4b-p2p / 128K / 6P+1D / hicache-ratio 2.0
- 收益侧 = shadow（决策不执行），损失侧 = full（实际 transfer）

## 关键结论（已记录）

- 命中率：shadow 56.16% / full 54.92%（目标 86.2%）
- transfer 成功率 1.6%（14 成功 / 893 失败），失败全因 source residency probe failed（device=0 host=0）
- 根因：mamba 模型（qwen3.5-4b）历史 full-KV 被 write_through demote 到 host，而 HICACHE_DIRECT 不支持 mamba state → 主动 transfer 机制性不可行（详见 ft_local/主动干预transfer失败-根因调查与根本矛盾.md）
