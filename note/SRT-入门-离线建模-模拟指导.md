# SRT 项目入门：主动 KV 干预的离线建模与模拟

> 目标：读懂项目在做什么，然后接手两件事——
> **① 用真实实验数据建离线收益/损失模型；② 在模拟环境里复现并校准存储分层问题。**

---

## 0. 一页速览

- **项目一句话**：LLM 推理服务里，缓存亲和路由能提高命中、但会把负载压到少数机器上；我们想做「主动 KV 干预」把热点缓存放平，结果在真实环境里发现——**对混合 Mamba 模型，主动干预的传输机制根本走不通**。
- **要做的**：不用碰 GPU 集群，只做两件事：
  1. 用 `experiments/offline_modeling_dataset_full.tar.gz`（脱敏日志 + 请求元数据，`hash_ids` 由负责人给）建**离线模型**，量化「主动干预到底值不值」；
  2. 用模拟器（`ft_local/kvsim_multipod.py`）**复现**「主动干预收益≈0」这个真实结论，关键是给模拟器补上它现在缺失的**存储分层**。
- **核心概念（先记住）**：主动干预能不能做，不取决于「预测得准不准」，而取决于「KV 此刻在哪个介质、这个介质能不能被传走」。

---

## 1. 项目背景：要解决什么问题

### 1.1 LLM 推理的成本在哪里

一个 LLM 请求 = prefill（读整段 prompt，算一遍）＋ decode（一个 token 一个 token 往外吐）。prompt 越长，prefill 越贵。生产里 prompt 中位数能到 7 万 token，prefill 是成本大头。

### 1.2 前缀缓存：省 prefill 的关键

很多请求共享同样的开头（system prompt、工具定义、多轮历史）。这段开头的 KV（Key/Value 缓存，注意力计算要用）**只取决于这段文字本身**，跟后面的内容无关。所以可以把这段 KV 存下来，下个请求开头相同就**直接复用、跳过这部分 prefill**。这就是「前缀缓存」。

引擎把 token 按页（如 512 token 一页）切块，每块算一个哈希，哈希链 = 前缀路径。请求来了，沿哈希链往下匹配，能匹配多深，就能省多少 prefill。

### 1.3 缓存亲和路由的两难

服务是多实例（多 pod）的，每个 pod 各存各的缓存。路由器要把请求分给某个 pod：

- **亲和路由**：把请求发给「已经缓存了它前缀」的 pod → 命中高，但同类请求全堆到同一个 pod，正反馈越滚越偏（实测负载不均 ≈ 随机分配的 **3.73 倍**）；
- **均衡路由**：发给最闲的 pod → 均衡，但命中低，prefill 全量重算。

两难：命中高 = 负载失衡，负载均衡 = 命中低。

### 1.4 主动 KV 干预的思路

既然亲和的价值来自「pod 上有缓存」，那为什么不**主动**把热点前缀的 KV 复制到多个 pod，让亲和路由自然把负载摊开？这就是「主动 KV 干预」。

我们最初按这个思路做了三种机制（§2），但最终发现（§3）：**在混合 Mamba 模型上，这个前提是假的。**

---

## 2. 我们的方案与演进

### 2.1 三种干预机制

| 机制 | 触发时机 | 做什么 | 解什么 |
|---|---|---|---|
| **预测式复制**（eager） | 需求出现前 | 按历史热度复制热链 | 通用热点 |
| **模板广播**（broadcast） | 需求出现前 | 把共享 system prompt 铺到所有 pod 并 pin 住 | 浅层高频复用 |
| **按需召回**（recall） | 需求出现后（miss 那刻） | 从持有者拉缺失前缀 | 深层稀疏复用 |

三者共享同一个「决策器 → 评估器 → 执行器」框架，底层都靠同一个 P2P KV 传输端点（`/experimental/p2p_kv_transfer`，无请求驱动、worker 间直传）。

### 2.2 从「时机问题」到「介质问题」

我们一度以为核心是「什么时候干预」——预测式复制 vs 广播 vs 召回。真实集群跑出来：**broadcast 传输成功率 0%，proactive 1.57%**，而且决策侧（发出去的 token/hash）验证完全正确。

排除掉「预测错」「时机不对」「索引过期」之后，根因落在**存储介质**上（§3）。这才是你们建模要抓的重点。

---

## 3. 核心问题：存储分层（必读）

> 这是整个项目最有意思、也最难建模的地方。先读 `主动干预transfer失败-根因调查与根本矛盾.md` §4–§6。

### 3.1 KV 存在哪：对混合 Mamba 模型有三层

我们的模型是 hybrid SSM（qwen3.5-4b，32 层 = 24 层 linear_attention ＋ 8 层 full_attention）。KV 状态分三层：

| 层 | 介质 | 内容 | 大小（实测） |
|---|---|---|---|
| device full-KV | GPU 显存 | attention 层的 K/V | ~29K token（≈一个请求） |
| host full-KV | 内存 | attention 层 K/V 的备份 | ~905K token |
| mamba host state | 内存 | linear_attention 层的递归状态 | （在 905K 内） |

### 3.2 write_through 把历史 KV 降级

引擎开 `--hicache-write-policy write_through`：一个请求 prefill 完，它的 full-KV **立刻从 device demote 到 host**。所以 **device 里永远只剩「当前正在 prefill 的请求」，历史缓存（包括热点模板）的 full-KV 全在 host**。

### 3.3 为什么主动 transfer 两条路都堵死

传输端点只有两条路：

1. **DEVICE 模式**（GPU 直传）：要求 source 的 **device 里有完整 full-KV** → 历史缓存的 full-KV 已被 demote → `device=0` → 失败。
2. **HICACHE_DIRECT 模式**（host 直传）：要求 `not requires_mamba()` → hybrid SSM 的 `requires_mamba=True` → **被拒**。

**矛盾链条**：主动干预要搬的是「历史热点前缀」→ 它的 full-KV 在 host（不在 device）→ DEVICE 搬不了；host 直传又不支持 mamba。**两条路全死。**

> 关键对比：对**纯 attention 模型**，HICACHE_DIRECT 可用（`requires_mamba=False`），主动干预还能走 host 直传（慢但能通）。所以「主动干预机制性失败」是**混合模型特有**的死锁。

### 3.4 一句话总结（导师要的「亮点」）

> 主动 KV 干预能不能做，不取决于「预测得准不准」，而取决于「KV 此刻在哪个介质、这个介质能不能被 transfer 路径搬走」。混合 Mamba 模型下，write_through 把历史 full-KV 全降级到 host，而 host 直传不支持 mamba、device 直传又要求 KV 在 device——**主动干预的前提在机制上不成立**。

### 3.5 这对建模意味着什么（你们任务的源头）

现有的离线回放/模拟器（`kvsim_multipod.py` 等）把 KV 当成「一块可传输的块」，**完全没有存储分层**。所以它们会高估主动干预的收益——真实环境里收益是 ≈0，模拟器却算出正收益。你们要做的最重要的一件事，就是**把 §3 的存储分层建进模型/模拟器里，让它能复现「主动干预 ≈0」这个真实结论**。

---

## 4. 你们要做的两件事

### 4.1 离线建模（用真实日志）

回答：**在什么条件下，主动 KV 干预才有正净收益？** 产出是一个「收益 − 损失」模型：

- **收益** = 省下的 prefill（命中提升 × 单 token 重算成本）；
- **损失** = 传输成本（wire 字节 × 单字节传输成本）＋ 误搬（传了没用的）＋ 驱逐（副本挤掉了别的内容）；
- **第一层闸门（关键）**：存储分层决定的「可传输性」——如果 KV 不在可传介质上，收益直接为 0，后面都不用算。

### 4.2 模拟环境测试（用回放模拟器）

回答：**把存储分层加进模拟器后，各策略的真实收益是多少？** 产出是策略对比表（AFF / REQ / 广播 / 召回 …），并让它对齐真实实验的两个硬数字：

- 主动 transfer 成功率 ≈ 0%（broadcast 0%、proactive 1.57%）；
- 干预组 vs 不干预组命中率无差异（56.16% vs 54.92%，差异在噪声内）。

---

## 5. 数据：`experiments/offline_modeling_dataset_full.tar.gz`

脱敏后的真实集群日志 + 请求元数据（**完整版 29.9 MB**；另有精简版 `offline_modeling_dataset.tar.gz` 692 KB，只含日志）。已脱敏：worker URL → p0..p5/d0，sub_uin → 哈希假名。**请求正文（messages/tools）是敏感数据，不在数据包里**；模拟所需的 `hash_ids` 由负责人单独提供（见 §6.1）。解压：

```bash
cd experiments
tar -xzf offline_modeling_dataset_full.tar.gz -C /tmp/kvprefetch_full   # 解到独立空目录
```

### 5.1 数据文件与字段

| 文件 | 大小 | 是什么 | 关键字段 |
|---|---|---|---|
| `hash_ids`（负责人提供） | — | 548 条请求的块哈希链（**模拟的输入**） | 对齐 `load_trace` 的 `hash_ids` |
| `request_meta.json` | 69 KB | 每条请求的会话结构 + **目标命中标注** | `idx` `sub_uin` `start_time` `n_tokens` `cached_target` |
| `per_request.jsonl` | 85 KB | 逐请求端到端结果（实际） | `idx` `prompt_tokens` `cached_tokens` `completion_tokens` `cache_hit_ratio` `ttfb` `ttlb` |
| `routing.jsonl` | 255 KB | 路由决策 | `reason` `matched_tokens_est` `hit_count` `selected` |
| `transfer.jsonl` | 492 KB | 传输结果 | `ok` `transferred_tokens` `elapsed_ms` `bandwidth_tokens_per_s` `fail_message` `reason` |
| `decisions.jsonl` | 18 MB | 决策器日志 | `msg` `hit_count` `depth` `benefit` `cost` `executed` |

### 5.2 关键数字（README 口径）

- 548 条真实 agent 请求（含 tools），单用户多轮，**156 种 system prompt 变体**交错回放，c=1；
- 模型 qwen35-4b-p2p，128K，6P+1D，hicache-ratio 2.0；
- **命中率**：shadow 56.16% / full 54.92%；**目标命中率（天花板）86.2%**；
- **transfer 成功率 1.6%**（14 成功 / 893 失败），失败全因 `source residency probe failed (device=0 host=0)`。

### 5.3 最重要的新东西：`cached_target`（目标命中天花板）

`request_meta.json` 里每条请求带 `cached_target`，含义是「这条请求在理想前缀缓存下**应该命中多少 token**」（由模板复用结构算出，与 `n_tokens` 之比 = 86.2%）。

它给了建模一个明确的**天花板**：

- **目标命中 86.2%** = 前缀缓存理论上能省的上限；
- **实际命中 ~56%** = 纯被动 prefix cache 拿到的东西；
- **中间那 ~30 个百分点的差距** = 主动干预的「潜在空间」。

而存储分层问题（§3）的结论恰恰是：**这 30 个百分点的空间，主动干预填不了**（transfer 走不通）。这是你们建模要定量回答的核心问题。

### 5.4 数据之间的关系（容易搞混，注意）

- `request_meta.json` + **负责人提供的 `hash_ids`** = **输入 trace**（回放/模拟的原料）。前者是时间戳 + token 数 + 目标命中，后者是块哈希链。
- `per_request.jsonl` = **实际结果**（请求级真相：命中多少、多快），是建模的「标签」。
- `routing.jsonl` / `decisions.jsonl` / `transfer.jsonl` = **系统内部日志**（路由怎么选、决策器怎么想、传输成没成），是解释「为什么」的中间证据。

> 注意：请求**正文（messages/tools）不提供**（敏感数据）。模拟所需的 `hash_ids` 由负责人单独给，你**不需要、也不应该**接触或重新渲染原文——之前项目踩过「渲染不一致导致 hash bug」的坑，所以哈希由负责人统一做、保证对齐 sglang。

## 6. 已有模拟代码

### 6.1 `ft_local/kvsim_multipod.py`（主模拟器，你们要改它）

多 pod 前缀缓存回放模拟器：块 = 512 token = 1 page，前缀树 + leaf-LRU，负载 = 最近 60s 的 miss token。支持：

- **路由**：`AFF`（纯亲和）/ `LEAST`（最闲）/ `REQ`（有界亲和 + 请求驱动复制）；
- **主动策略**：`Oracle`（未来 W 秒需求排序，副本上限 + 深度限制 + 预算）、`stable_broadcast`（最常出现的首块预置到所有实例）。

**输入格式**（`load_trace`）：每行 `{timestamp, input_length, output_length, hash_ids}`。数据来源有两个：
- Mooncake Kimi 公开 trace（`data/mooncake/{wl}_trace.jsonl`，自带 hash_ids）；
- **本项目自己的 trace**：`request_meta.json`（提供 `start_time`/`n_tokens`）+ **负责人提供的 `hash_ids`** 合并而成。

**注意**：请求正文是敏感数据，**不提供**。`hash_ids` 由负责人用对齐 sglang 的渲染 + 哈希统一算好给你，你直接把 `hash_ids` 与 `request_meta.json` 的 `start_time`/`n_tokens` 拼成 trace 即可，**不要自己重新渲染原文**（之前项目踩过「渲染不一致导致 hash bug」的坑）。

```bash
python ft_local/kvsim_multipod.py conversation base      # AFF/LEAST/REQ 基线
python ft_local/kvsim_multipod.py conversation idea2     # 广播
python ft_local/kvsim_multipod.py conversation all       # 全部
```

### 6.2 `intervention/replay.py`（可选，更细的影子仿真）

复用在线成本模型做离线回放，数据源同样是 Mooncake trace，输出策略对比表。可作为参照，但本科生的重点是改 `kvsim_multipod.py`。

---

## 7. 任务拆解（按顺序做）

### 任务 1：读懂数据，复现「transfer 失败」结论（第 1 周）

写一个小脚本，从 `transfer.jsonl` 里统计：

1. 按 `reason`（broadcast/proactive/recall）分组的 `ok/rejected` 数与成功率；
2. `fail_message` 的分布（预期全是 `source residency probe failed ... device=0 host=0`）；
3. 成功那几次的 `requested_tokens` 分布（预期都是浅前缀 128/256）。

4. 用 `request_meta.json` 的 `cached_target` / `n_tokens` 标定**天花板**：目标命中率 86.2% vs 实际 56%，算出主动干预的「潜在空间」是多少个百分点。b

**产出**：一张表 + 一段话，说明「为什么 transfer 几乎全失败」，再补一句「主动干预的潜在空间有多大、为什么填不了」。这一步是为了确保你理解 §3 的存储分层。

### 任务 2：给 `kvsim_multipod.py` 加存储分层（第 2–3 周，核心）

> 输入用负责人给的 `hash_ids` + `request_meta.json` 拼成的 trace（或 Mooncake trace，见 §6.1）。

现在的模拟器里 `PodCache` 只有**一个** cache，KV 是「可传输块」。你要改成一个**两层结构**：

- **device 层**（小）：只留「当前正在 prefill 的请求」的 full-KV，插入后立刻 demote；
- **host 层**（大）：历史 full-KV（attention 层）＋ mamba state。

并且让「主动干预」的转移规则对齐真实引擎：

- 主动 transfer 只能搬 **device 里存在的 full-KV**（DEVICE 模式）；
- host 直传对 hybrid SSM 不可用（`requires_mamba=True`）；
- 于是：主动 transfer 的成功率 ≈ 0（因为历史热点前缀永远不在 device）。

**产出**：一个能复现「广播/proactive 传输成功率 ≈ 0、干预组命中率 ≈ 不干预组」的模拟器。跑出来的策略对比表要能解释为什么之前（没分层时）的收益是虚高的。

### 任务 3：建 transfer 可行性模型（第 3–4 周）

用 `transfer.jsonl` 做二分类（`ok` = 1，`rejected` = 0），特征用 `requested_tokens`、`reason`、`source`、`target`、`elapsed_ms`。

**目标不是追求分类精度，而是验证一个假设**：失败与「深度」「时机」「source 是谁」都无关，只与「source 的 device 里有没有 full-KV」有关（而后者恒为 false）。产出是一句可检验的结论：**在 write_through + hybrid SSM 下，主动 transfer 的可行性 ≈ 0，与调度策略无关。**

### 任务 4：建收益/损失模型（第 4–6 周，收口）

把「收益 − 损失」写成显式模型，用三个数据源标定：

- `per_request.jsonl` → 命中带来的 TTFT 收益（`cache_hit_ratio` 对 `ttfb` 的关系）；
- `decisions.jsonl` → 决策器估计的 `benefit`/`cost`（对照真实，看高估多少）；
- `transfer.jsonl` → 真实传输成本（`elapsed_ms`/`bandwidth`）与成功率。

**模型输出**：给定一批请求，预测「如果开主动干预，净收益是多少」。第一层就是 §3 的可行性闸门——对 hybrid SSM，直接输出 ≈0，与真实实验对齐。然后问一个更有意思的问题：**假设把 host 直传放开（支持 mamba）或把热前缀 pin 在 device，净收益会变成多少？** 这能定量回答「要花多大代价让主动干预变得可行」。

---

## 8. 环境与命令

```bash
# 纯 Python，无 GPU 依赖
python3 --version        # 3.8+ 即可

# 解压数据
cd experiments && tar -xzf offline_modeling_dataset_full.tar.gz -C /tmp/kvprefetch_full

# 跑模拟器（需要 Mooncake trace，见 §6.1）
python ft_local/kvsim_multipod.py conversation base
```

只有 `kvsim_multipod.py` / `intervention/replay.py` 用到标准库 + 简单数据结构，无第三方依赖。GPU、sglang、集群相关一律**不需要**，那是集群侧的事。

---

## 9. 验收标准与交付物

| 阶段 | 交付物 | 判据 |
|---|---|---|
| 任务 1 | 数据统计脚本 + 一张表 | 能复现「transfer 成功率 ≈0%、失败全因 device=0 host=0」 |
| 任务 2 | 改好的 `kvsim_multipod.py`（两层缓存） | 模拟器复现「主动干预收益 ≈0」，能解释之前虚高 |
| 任务 3 | 可行性模型 + 结论 | 结论是「可行性由介质决定，与调度策略无关」 |
| 任务 4 | 收益/损失模型 + 报告 | 能定量回答「放开 host 直传 / pin device 后净收益多少」 |

**最终要回答的三个问题**（写进总结报告）：

1. 为什么主动 KV 干预在混合 Mamba 模型上失败？（存储分层，代码级根因）
2. 离线模型之前为什么高估收益？（把 KV 当可传输块，漏了分层）
3. 要让主动干预可行，引擎要做什么改动、付出多大代价才划算？

---

## 10. 常见问题（FAQ）

**Q：我要不要懂 GPU / sglang / K8s？**
不用。你的工作全部是 Python 数据分析 + 纯 Python 模拟，不碰集群。

**Q：完整版数据集和 `kvsim_multipod.py` 用的数据什么关系？**
完整版数据集里是**日志**（decisions/routing/transfer/per_request，用来**建模**）+ **请求元数据**（request_meta.json）。模拟要用的 `hash_ids` 由负责人单独给（请求正文是敏感数据，不进包）。把 `request_meta.json` 的 `start_time`/`n_tokens` 与 `hash_ids` 拼起来就是模拟 trace。Mooncake 公开 trace 是另一份自带 hash_ids 的模拟数据。

**Q：存储分层建模，最关键的假设是什么？**
「历史热点前缀的 full-KV 永远在 host、不在 device」。你可以在任务 2 里把 device 层设成「只留当前请求」，这是对齐真实引擎 write_through 的关键。

**Q：如果我发现模拟器复现不了 0% 成功率怎么办？**
检查两点：① device 层是否真的「插入即 demote」；② 主动 transfer 是否错误地搬了 host 层。这两点是真实失败的直接原因，模拟器必须忠实还原。

---

> 有任何不确定的地方，先读 `ft_local/主动干预transfer失败-根因调查与根本矛盾.md`（尤其 §4、§6 的代码级根因），再问负责人。不要凭猜写模型。
