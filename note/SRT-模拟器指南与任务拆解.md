# SRT 模拟器指南与任务拆解

> 本文件讲三件事：
> ① 模拟器 `kvsim_multipod.py` 逐块解析；② 存储分层改造要改哪里（精确到函数和行）；③ 阅读材料与细化的任务拆解。
> 动手改代码前，先读完本文件 + `主动干预transfer失败-根因调查与根本矛盾.md`。

---

## 1. 模拟器 `kvsim_multipod.py` 完全解析

文件共 305 行，纯标准库（json/math/random/sys/collections/bisect），无第三方依赖。默认块 = 512 token = 1 page（Mooncake trace 口径）；**本项目 `trace.jsonl` 的块 = 64 token（mamba chunk），用本项目 trace 时要把第 11 行 `BLOCK` 改成 64**（见 §2.4）。

### 1.1 逐块结构

| 行号 | 内容 | 作用 |
|---|---|---|
| 11–13 | 常量 `BLOCK=512`（**本项目 trace 改 64**）`PAGE_BYTES=14MiB` `GIB` | 每块 token 数、每页字节、单位换算 |
| 16–24 | `load_trace(path)` | 读 `{timestamp, input_length, output_length, hash_ids}`，按时间排序（`cached_target` 等额外字段被忽略） |
| 27–99 | `class PodCache` | **单个 pod 的前缀树缓存**（核心，要改这里） |
| 102–110 | `gini(xs)` | 基尼系数，衡量负载不均 |
| 113–121 | `random_gini(N,total)` | 随机分配的期望基尼，作「失衡度」归一化分母 |
| 124–135 | `potential_ceiling(reqs)` | 无限缓存下的 saved 上界（天花板） |
| 138–263 | `run(...)` | **主模拟循环**（路由 → serve → 主动干预） |
| 266–304 | `__main__` | CLI，跑各策略组合 |

### 1.2 `PodCache` 内部（27–99 行）

```python
class PodCache:
    def __init__(self, capacity_pages, infinite=False):
        self.nodes = {}              # nid -> [parent_nid, block_hash, last_access]
        self.children = defaultdict(dict)  # (parent_nid, block_hash) -> nid
        self.roots = {}              # block_hash -> nid（根节点）
        self.size = 0                # 当前块数
        self.next_id = 0
```

- `hit_depth(blocks, t)`（37–52）：沿块哈希链往下匹配，返回能匹配多深，并顺带刷新命中节点的 `last_access`。
- `insert(blocks, t)`（54–71）：逐块建节点；超容量时调 `_evict()`。
- `_evict()`（73–90）：**leaf-LRU** 淘汰——遍历找「最久未访问的叶子节点」删掉，循环到 `size <= cap`。
- `contains(blocks)`（92–99）：判断整条链是否存在。

**关键点**：这是一个**单层** cache，块要么在、要么不在，没有「device / host」之分。这正是它高估主动干预收益的根源（见 §2.1）。

### 1.3 `run` 主循环的三步（170–256 行）

对每条请求 `(t, il, ol, h)` 做三步：

```
1) routing（171–207）：决定发给哪个 pod
   AFF  = 选 hit_depth 最深的 pod（平手选负载低的）
   LEAST= 选 60s 内 miss tokens 最少的 pod
   REQ  = 先按 AFF 选；若源负载 > theta × 最闲，改投最闲 + 把前缀插进目标（请求驱动复制）

2) serve（209–219）：命中 d = hit_depth(h)，saved = min(d*512, il)，
   insert(h) 进该 pod，记录 miss=il-saved 进 60s 负载窗口

3) proactive（221–256）：仅当 policy=="Oracle"
   看未来 W 秒的需求，选得分最高的链 ch，若副本数 < cap2，插进最闲的非持有者
```

### 1.4 指标口径

| 指标 | 含义 | 计算 |
|---|---|---|
| `saved` | 评价窗内省下的 prefill token | Σ min(hit_depth×512, il) |
| `token_hit` | 命中率 | saved / input |
| `skew` | 负载失衡度 | gini(pod_req) / random_gini（≈1 表示随机，>1 表示失衡） |
| `wire_GiB` | REQ 请求驱动复制的传输量 | Σ 前缀块 × 14MiB |
| `proactive_wire_GiB` | 主动干预的传输量（评价窗内） | 同上 |

---

## 2. 存储分层改造：改哪里、改成什么

### 2.1 为什么现在会高估收益（先理解这个）

看 **252 行**：

```python
pods[target].insert(ch[:eff], t)   # Oracle 主动策略：把热链插进 target
```

这行是「**凭空把链插进 target 的 cache**」，等于默认「transfer 一定成功」。而真实环境里（根因调查文档 §3）transfer **几乎全失败**——因为要搬的热点前缀的 full-KV 已被 write_through demote 到 host，device 直传搬不到、host 直传又不支持 mamba。

所以单层模型算出「主动干预有正收益」，真实环境是 ≈0。**改造目标就是让模拟器忠实还原这个失败。**

### 2.2 改造方案（推荐，最小改动）

把「device / host 两层」建模成：**device 是一个容量极小的 PodCache，host 是原来的 PodCache**；`insert` 时新块先进 device、device 满了就把最老的 demote 到 host；主动 transfer **只允许从 device 搬**。

具体三步：

**第 1 步：`PodCache` 加一个 device 层（改 27–99 行）**

- 给 `PodCache` 加 `self.device`（一个容量 ≈ 一个请求的小 cache，可用 `PodCache(device_cap)` 复用现有逻辑）和 `self.host`（原来的结构）。
- 保留对外接口不变（`hit_depth`/`insert`/`contains`），内部改为「先查 device 再查 host」。

**第 2 步：`insert` 加 demote 语义（改 54–71 行）**

- 新块进 device；device 超 `device_cap` 时，把最老的块移到 host（模拟 write_through）。
- 效果：device 里永远只剩「最近一个请求」的 full-KV，历史缓存（含热点模板）全在 host——对齐真实引擎（device 实测 ~29K token ≈ 一个请求，host ~905K）。

**第 3 步：主动 transfer 加源检查（改 252 行 + run 返回值）**

- transfer 前，用 **source（holder）的 device** 查 `hit_depth`，若 device 命中深度 < 要搬的 `eff`，则 **transfer 失败**：不 insert，计入 `transfer_failed`。
- 因为 hybrid SSM 下 source 的 device 里没有历史热点前缀，这一步会让成功率 ≈ 0，从而复现真实结论。

### 2.3 具体改动点清单

| 位置 | 现在 | 改成 |
|---|---|---|
| `PodCache.__init__`（28–35） | 单层 nodes/children/roots | 加 `self.device`（小 PodCache）+ `self.host`（原结构），加 `device_cap` 参数 |
| `hit_depth`（37–52） | 只查一层 | 分别查 device、host，返回总深度 + device 深度（供 transfer 源判断） |
| `insert`（54–71） | 直接进单 cache | 先进 device，超 cap 则最老块 demote 到 host |
| `_evict`（73–90） | host 淘汰（保留） | host 照旧 leaf-LRU；device 用 demote 而非淘汰 |
| `run` 主动 transfer（252） | 直接 `insert(ch[:eff])` | 先查 source 的 device 命中，不足 `eff` 则失败、计入 `transfer_failed` |
| `run` 返回值（258–263） | 无 transfer 成败统计 | 加 `transfer_ok` / `transfer_failed` / 成功率 |
| `__main__`（271） | `N,C = 16, 585` | 加 `device_cap`（建议默认 57 page ≈ 29K token） |

> **性能坑（任务 2 会撞上）**：`_evict`（73–90 行）是 O(size) 朴素实现——每次淘汰遍历全部节点找最老叶子。块=64 时块数是块=512 的 8 倍，完整 548 条请求会慢到跑不动（实测前 50 条 + C=2000 三条路由约 6 秒，全量 548 条 + C=14000 直接超时）。加存储分层时**建议顺带把淘汰改成 O(log size)**（用堆/有序结构维护最老叶子）。快速演示可直接跑同目录的 `srt_quickstart.py`（已内置前 50 条子集）。

### 2.4 验收判据（改前改后对比）

改**之前**先跑基线并记下来。**用本项目 `trace.jsonl` 时：先把第 11 行 `BLOCK = 512` 改成 `BLOCK = 64`，再把 `__main__` 里硬编码的 `path`（`data/mooncake/{wl}_trace.jsonl`）指到 full 包的 `trace.jsonl`**：

```bash
python ft_local/kvsim_multipod.py conversation base      # AFF / LEAST / REQ
python ft_local/kvsim_multipod.py conversation idea2     # 广播
python ft_local/kvsim_multipod.py conversation idea3     # Oracle 主动
```

改**之后**，同一命令应满足：

1. **AFF / LEAST / REQ 基线基本不变**（这三个路由不靠主动 transfer，分层不应影响它们）；
2. **Oracle 主动策略的 `proactive_wire_GiB` 接近 0、`transfer_failed` 占绝大多数**，`token_hit` 与不干预的 REQ 几乎一样——即「主动干预收益 ≈ 0」；
3. 能写清楚一句解释：**为什么加了 device/host 分层后，主动干预从「正收益」变成「≈0」**。

---

## 3. 阅读材料

### 3.1 内部文档

| 文档 | 读它理解什么 |
|---|---|
| `主动干预transfer失败-根因调查与根本矛盾.md` | **存储分层根因**（改造模拟器的依据，必读） |

### 3.2 学术论文

| 论文 / 资源 | 理解什么 |
|---|---|
| **vLLM: Efficient Memory Management for LLM Serving with PagedAttention**（SOSP 2023） | KV cache 分页、为什么 KV 是瓶颈 |
| **SGLang: Efficient Execution of Structured Language Model Programs**（NeurIPS 2024） | RadixAttention、前缀树、块哈希链（本项目前缀缓存的来源） |
| **vLLM Automatic Prefix Caching**（官方文档，搜标题） | 前缀缓存怎么复用、命中怎么算 |
| **Mooncake: A KVCache-centric Disaggregated Architecture**（FAST 2025） | 分布式 KV、PD 分离、L3 存储分层（本项目环境的背景） |

### 3.3 进阶（可选，做收益/损失模型时再看）

- **CacheBlend** / **LMCache**：KV 缓存的复用与跨实例分发，理解「传输成本 vs 重算成本」的权衡。
- 经典缓存理论里的 **eager vs lazy prefetch**：理解「预测式复制 vs 按需召回」背后的取舍（本项目 §2.2 的学术母题）。

---

## 4. 任务拆解（细化版）

### 阶段 0：热身

- 解压 full 包，先跑 **`srt_quickstart.py`**（同目录，一键复现 transfer 失败率 / 天花板 / kvsim 基线三个结论），确认环境 OK；再自己跑 `kvsim_multipod.py`（用 `trace.jsonl`，先把 `BLOCK` 改 64；或 Mooncake trace 块=512）。
- 能解释输出里 `saved / hit / skew / pwire` 各是什么。

### 阶段 1：读懂数据 + 复现失败结论
- 从 `transfer.jsonl` 统计 ok/rejected 成功率（预期 ≈1.6%，失败全因 `source residency probe failed`）。
- 用 `request_meta.json` 的 `cached_target`/`n_tokens` 标定天花板（目标 86.2% vs 实际 ~56%，差距 = 主动干预的潜在空间）。

### 阶段 2：读懂模拟器 + 跑基线
- 逐块读懂 `PodCache` / `run` 的三路由 + Oracle 主动策略（对照 §1）。
- 跑 `base` / `idea2` / `idea3`，记录基线输出，作为改造后对照。

### 阶段 3：加存储分层
- 3a. `PodCache` 拆 device/host 两层 + demote 语义；
- 3b. 主动 transfer 加「source device 命中检查」；
- 3c. 加 transfer 成败统计；
- 3d. 复现「主动干预 ≈ 0」，写清为什么（对照 §2.4 验收判据）。

### 阶段 4：建 transfer 可行性模型
- 用 `transfer.jsonl` 做二分类，验证假设：**可行性由「source 的 device 里有没有 full-KV」决定，与深度/时机/source 是谁无关**。

### 阶段 5：建收益/损失模型
- 综合 `per_request`（收益）+ `decisions`（决策器估计）+ `transfer`（真实成本/成功率），建「收益 − 损失」模型。
- 第一层闸门 = §3 存储分层的可传输性；对 hybrid SSM 输出 ≈0。
- 再问：**放开 host 直传（支持 mamba）或把热前缀 pin 在 device，净收益变成多少？** 定量回答「要花多大代价让主动干预变得可行」。

---

## 5. 三个最终要回答的问题

1. 为什么主动 KV 干预在混合 Mamba 模型上失败？（存储分层，代码级根因）
2. 离线模型 / 单层模拟器之前为什么高估收益？（把 KV 当可传输块，漏了分层）
3. 要让主动干预可行，引擎要做什么改动、付出多大代价才划算？
