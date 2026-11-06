# 实验记录

> ### 阶段 0：热身
>
> - 解压 full 包，先跑 **`srt_quickstart.py`**（同目录，一键复现 transfer 失败率 / 天花板 / kvsim 基线三个结论），确认环境 OK；再自己跑 `kvsim_multipod.py`（用 `trace.jsonl`，先把 `BLOCK` 改 64；或 Mooncake trace 块=512）。
> - 能解释输出里 `saved / hit / skew / pwire` 各是什么。
>
> ### 阶段 1：读懂数据 + 复现失败结论
>
> - 从 `transfer.jsonl` 统计 ok/rejected 成功率（预期 ≈1.6%，失败全因 `source residency probe failed`）。
> - 用 `request_meta.json` 的 `cached_target`/`n_tokens` 标定天花板（目标 86.2% vs 实际 ~56%，差距 = 主动干预的潜在空间）。
>
> ### 阶段 2：读懂模拟器 + 跑基线
>
> - 逐块读懂 `PodCache` / `run` 的三路由 + Oracle 主动策略（对照 §1）。
> - 跑 `base` / `idea2` / `idea3`，记录基线输出，作为改造后对照。
>
> ### 阶段 3：加存储分层
>
> - 3a. `PodCache` 拆 device/host 两层 + demote 语义；
> - 3b. 主动 transfer 加「source device 命中检查」；
> - 3c. 加 transfer 成败统计；
> - 3d. 复现「主动干预 ≈ 0」，写清为什么（对照 §2.4 验收判据）。
>
> ### 阶段 4：建 transfer 可行性模型
>
> - 用 `transfer.jsonl` 做二分类，验证假设：**可行性由「source 的 device 里有没有 full-KV」决定，与深度/时机/source 是谁无关**。
>
> ### 阶段 5：建收益/损失模型
>
> - 综合 `per_request`（收益）+ `decisions`（决策器估计）+ `transfer`（真实成本/成功率），建「收益 − 损失」模型。
> - 第一层闸门 = §3 存储分层的可传输性；对 hybrid SSM 输出 ≈0。
> - 再问：**放开 host 直传（支持 mamba）或把热前缀 pin 在 device，净收益变成多少？** 定量回答「要花多大代价让主动干预变得可行」。

## 数据集整理

### 1. request_meta.json

> 请求的基本信息和目标命中标注。

文件是一个 JSON 数组，共 **548 条记录**，每条对应一个请求。

| 字段            | 保存的数据                                                   |
| --------------- | ------------------------------------------------------------ |
| `idx`           | 请求编号                                                     |
| `sub_uin`       | 脱敏后的用户标识                                             |
| `start_time`    | 原始数据中的请求开始时间                                     |
| `n_tokens`      | 原始请求的输入 token 数                                      |
| `cached_target` | 该请求的目标命中 token 数标注<br />(理想情况下预计能够命中的前缀 token 数 |

例如，第 0 条请求记录了：

```
idx = 0
start_time = 2026-05-13T14:33:00+08:00
n_tokens = 27601
cached_target = 25664
```

这个文件没有请求正文，也没有前缀哈希链。当前数据中，`sub_uin` 只有一个不同的取值。

### 2. trace.jsonl

> 供模拟器回放的请求输入

共 **548 行**，每行一个请求

| 字段            | 保存的数据                                                   |
| --------------- | ------------------------------------------------------------ |
| `timestamp`     | 原始请求时间，用于确定回放时间和顺序                         |
| `input_length`  | 请求输入的 token 数                                          |
| `output_length` | 请求输出的 token 数；当前文件中全部为 0，输出长度保存在 `per_request.jsonl` 中 |
| `hash_ids`      | 按顺序排列的前缀块哈希链                                     |
| `cached_target` | 目标命中 token 数标注                                        |

其中最关键的是 `hash_ids`。模拟器通过比较请求之间的哈希链，识别它们共享了多长的前缀。

本项目的哈希块大小为 **64 tokens**，每条请求的哈希数量为：

$ \left\lceil \frac{input\_length}{64}\right\rceil $

例如，第 0 条请求有 27,601 个输入 tokens，对应 432 个哈希块。

### 3. routing.jsonl

> Gateway 的路由选择记录。

共 **1,739 行**，每行是一条路由日志。

| 字段                 | 保存的数据                   |
| -------------------- | ---------------------------- |
| `ts`                 | 路由日志时间                 |
| `reason`             | 路由选择的原因               |
| `matched_tokens_est` | 路由器估计可命中的 token 数  |
| `hit_count`          | 路由日志记录的命中计数       |
| `selected`           | 最终选择的 Worker，例如 `p0` |
| `matched_rate_pct`   | 路由器估计的命中百分比       |

当前文件中有两种 `reason`：

| `reason`         | 数量  |
| ---------------- | ----- |
| `no_cache_owner` | 1,712 |
| `cache_overlap`  | 27    |

### 4. decisions.json

> 主动干预决策器的评估与动作记录

共 **79,312 行**。它包含候选评估、具体决策及一轮评估的汇总，因此每一行不一定对应一次实际传输。

| 字段           | 保存的数据                                                   |
| -------------- | ------------------------------------------------------------ |
| `ts`           | 决策日志时间                                                 |
| `msg`          | 日志类型                                                     |
| `hit_count`    | 决策器记录的候选前缀命中／热度计数<br />（由决策器记录的索引来计算的命中数） |
| `depth`        | 候选前缀深度，按 token 长度记录                              |
| `benefit`      | 决策器估计的收益值                                           |
| `cost`         | 决策器估计的成本值                                           |
| `source`       | 计划使用的源 Worker                                          |
| `target`       | 计划使用的目标 Worker                                        |
| `executed`     | 是否执行该决策的标记                                         |
| `template_hit` | 广播决策中的模板命中计数                                     |
| `chains`       | 一轮评估记录的候选链数量                                     |
| `placements`   | 一轮评估记录的放置方案数量                                   |

当前包含三类日志：

| `msg`                | 数量   | 记录内容                 |
| -------------------- | ------ | ------------------------ |
| `proactive-decide`   | 65,653 | Proactive 候选的具体决策 |
| `broadcast-decide`   | 9,687  | Broadcast 的具体决策     |
| `proactive evaluate` | 3,972  | 一轮主动评估的汇总       |

不同类型只填写相关字段，其余字段为 `null`

这个文件主要回答：

> 决策器选中了什么候选？准备从哪里复制到哪里？它当时估计的收益和成本是多少？

`executed=true` 是决策执行标记，**传输是否成功仍要查看 `transfer.jsonl`**。`benefit` 和 `cost` 也是决策器估计值

### 5. transfer.jsonl

> 主动传输尝试的执行结果

共 **1,508 行**，每行对应一次传输尝试。

| 字段                     | 保存的数据                                        |
| ------------------------ | ------------------------------------------------- |
| `ts`                     | 传输日志时间                                      |
| `ok`                     | 这次传输是否成功                                  |
| `requested_tokens`       | 请求传输的前缀长度                                |
| `transferred_tokens`     | 实际成功传输的 token 数                           |
| `elapsed_ms`             | 这次尝试记录的耗时，单位为毫秒                    |
| `bandwidth_tokens_per_s` | 记录的传输速率，单位为 tokens/s                   |
| `source`                 | 源 Worker                                         |
| `target`                 | 目标 Worker                                       |
| `reason`                 | 发起传输的干预类型，例如 `broadcast`、`proactive` |
| `fail_message`           | 失败信息，成功记录中为空字符串                    |

成功记录的例子：

```
reason = proactive
source = p3
target = p0
requested_tokens = 128
transferred_tokens = 128
elapsed_ms = 6
ok = true
```

### 6. per_request.jsonl

> 每个请求在实际集群中的执行结果

共 **548 行**，每行对应一个请求。

| 字段                | 保存的数据                            |
| ------------------- | ------------------------------------- |
| `idx`               | 请求编号                              |
| `prompt_tokens`     | 实测输入 token 数                     |
| `cached_tokens`     | 实测缓存命中 token 数                 |
| `completion_tokens` | 实际生成的输出 token 数               |
| `cache_hit_ratio`   | 该请求的缓存命中比例                  |
| `ttfb`              | 首字节响应耗时，即 TTFB               |
| `ttlb`              | 最后一个字节返回的耗时，即 TTLB       |
| `error`             | 请求错误信息；当前 548 条均为空字符串 |

这里的`cached_tokens`是实际的命中数

实际命中率使用：

$ \frac{\sum cached\_tokens}{\sum prompt\_tokens} $



总结：

| 需要了解的问题                           | 使用的文件          |
| ---------------------------------------- | ------------------- |
| 原始请求的基本信息和目标标注是什么？     | `request_meta.json` |
| 模拟器要回放哪些请求，它们共享哪些前缀？ | `trace.jsonl`       |
| Gateway 把请求发给了哪个 Pod？           | `routing.jsonl`     |
| 主动策略为什么选择这次复制？             | `decisions.jsonl`   |
| 复制有没有成功，耗时多少？               | `transfer.jsonl`    |
| 请求最终命中了多少、响应多快？           | `per_request.jsonl` |

注意几种不同的命中数：

| 信息                   | 反映什么                                 |
| ---------------------- | ---------------------------------------- |
| 路由／决策侧的匹配计数 | 系统根据自身索引观察到的匹配情况         |
| `matched_tokens_est`   | 根据索引估计的可命中 token 数            |
| `cached_tokens`        | Worker 执行请求后报告的实际命中 token 数 |

## stage 0

启动模拟器，对前50条请求进行模拟，结果为：

| 路由  | 命中率 Hit | Saved             | 请求倾斜度 Skew |
| ----- | ---------- | ----------------- | --------------- |
| AFF   | 35.40%     | 约 80.7 万 tokens | 5.12            |
| LEAST | 32.41%     | 约 73.8 万 tokens | 0.82            |
| REQ   | 43.58%     | 约 99.3 万 tokens | 0.78            |

| 指标    | 含义                                                         |
| ------- | ------------------------------------------------------------ |
| `saved` | 缓存命中省去的 Prefill token 数                              |
| `hit`   | `saved / 输入 token 总数`，即命中率                          |
| `skew`  | 各 Pod 请求数的 Gini，除以相同请求数量下随机分配的参考 Gini，接近1说明请求分布接近于随机分布 |
| `pwire` | 评价窗口内主动复制产生的数据传输量                           |

## stage 1

> 读取 transfer.json，统计传输情况。读取 `request_meta.json` 的 `cached_target`/`n_tokens` ，这是理想情况下的命中率，用于和实际情况对比

**1. 传输成功率**

| 类型      | 尝试数 | 成功 |  拒绝 | 成功率 |
| --------- | -----: | ---: | ----: | -----: |
| Broadcast |    615 |    0 |   615 |  0.00% |
| Proactive |    893 |   14 |   879 |  1.57% |
| 合计      |  1,508 |   14 | 1,494 |  0.93% |

这里每条记录表示一次传输尝试。548 条请求的执行过程中，策略可以多次发起复制，所以传输尝试数会超过请求数。

Broadcast 全部被拒绝；Proactive 只有 14 次成功。Proactive 组成功率约为 1.6%，全部主动传输的成功率为 **0.93%**。

**2. 失败信息**

1,494 条失败记录的 `fail_message` 相同：

```text
source residency probe failed: HTTP 400: {"rid":null,"http_worker_ipc":null,"success":false,"message
```

说明这些传输尝试在 **Source Residency Probe（源端驻留检查）阶段被拒绝**。Gateway 发起复制后，Source 需要重新确认自己是否具有要求的可传输前缀，这批尝试没有通过检查。

**3. 成功传输的长度**

| 请求传输长度 requested_tokens | 实际传输长度 transferred_tokens | 成功次数 |
| ----------------------------: | ------------------------------: | -------: |
|                           128 |                             128 |        9 |
|                           256 |                             256 |        5 |

成功的 14 次全部是 128 或 256 tokens 的浅前缀，而且实际传输长度都达到了请求长度。

**4. 目标命中率与实际命中率**

两份文件各有 548 条请求记录：

| 数据来源                      | 输入 token 总数 | 命中 token 总数 | 命中率 |
| ----------------------------- | --------------: | --------------: | -----: |
| `request_meta.json`：目标标注 |      26,196,338 |      22,570,688 | 86.16% |
| `per_request.jsonl`：实际结果 |      26,185,198 |      14,380,544 | 54.92% |

- `cached_target`：数据方预计理想情况下预计能够命中的前缀 token 数
- `cached_tokens`：本次集群实验实际命中的 token 数。

计算方式：

```text
目标命中率 = Σcached_target / Σn_tokens
           = 22,570,688 / 26,196,338
           = 86.16%

实际命中率 = Σcached_tokens / Σprompt_tokens
           = 14,380,544 / 26,185,198
           = 54.92%

目标与实际差值 = 86.16% - 54.92%
               = 31.24 个百分点
```

**总结**：主动传输绝大多数在源端检查阶段被拒绝，少量成功样本只覆盖浅前缀；实际 token 命中率为 54.92%。

## stage 2

### 模拟器架构

<img src="https://raw.githubusercontent.com/hhr2449/pictureBed/main/img/image-20261104180117030.png" alt="image-20261104180117030" style="zoom: 67%;" />

**1. 输入trace**

`load_trace`函数，输入trace的路径，保存为请求列表

每一条请求为`(t, il, ol, h)`

| 变量 | 含义                         |
| ---- | ---------------------------- |
| `t`  | 请求到达时间，模拟中使用毫秒 |
| `il` | 输入 token 数                |
| `ol` | 输出 token 数                |
| `h`  | 请求的 block 哈希链          |

**2. PodCache类**

表示一个pod的缓存，创建时可以指定容量上限`Cap`

内部被组织成一颗前缀树

| 变量       | 内容                                                         |
| ---------- | ------------------------------------------------------------ |
| `nodes`    | 字典，映射为：节点编号 → `[父节点编号, block_hash, 最后访问时间]` |
| `children` | 父节点下有哪些子节点                                         |
| `roots`    | 根部的 block hash 到节点编号的映射                           |
| `size`     | 当前保存的 block 数                                          |
| `cap`      | 缓存容量上限                                                 |
| `next_id`  | 给新节点分配编号                                             |

每个节点都是一个block，保存block hash

支持四种操作：前缀匹配深度、插入、淘汰、判断前缀路径是否存在

| 方法                   | 做什么                                     |
| ---------------------- | ------------------------------------------ |
| `hit_depth(blocks, t)` | 从开头沿前缀匹配，返回连续命中的 block 数  |
| `insert(blocks, t)`    | 插入前缀，复用已有节点，为缺失部分创建节点 |
| `_evict()`             | 容量超限时淘汰缓存                         |
| `contains(blocks)`     | 判断一整条前缀路径是否存在                 |

淘汰策略是 leaf-LRU

**3. 指标计算**

1. gini
2. random_gini
3. potential_ceiling

**4. 主流程run**

对每条请求依次完成“路由、服务、主动干预”

维护两个状态：

```
pods   # 每个 Pod 的缓存
loads  # 每个 Pod 最近处理请求的 miss 工作量记录
```

“负载”定义为：

$Load_i(t) = \text{Pod }i\text{ 在最近 60 秒内累计的 miss-prefill tokens} $



**4.1 路由**

三种策略

| 路由  | 选择方法                                                     |
| ----- | ------------------------------------------------------------ |
| AFF   | 优先选择前缀命中最长的 Pod；命中相同时选择负载更低的         |
| LEAST | 选择最近 60 秒 miss 工作量最低的 Pod                         |
| REQ   | 先按 AFF 选择；如果亲和 Pod 过载，则改投较空闲 Pod，并尝试复制前缀 |

AFF会调用`hit_depth`找命中前缀最长的pod，然后路由过去

LEAST会根据pod_load进行选择

REQ先按照AFF选择，如果过载（$Load_{\text{亲和 Pod}} > \theta \times \min(Load_{\text{其他 Pod}}) $），则选择负载最小的pod
如果选择的pod没有亲和pod中保存的前缀cache(使用contains判断)，则复制过去（直接insert）

**4.2 服务**

模拟Prefill过程，统计：命中多少，需要重算多少

然后更新状态：

- 将 `saved` 累计到评价指标；
- 给执行 Pod 的请求数加一；
- 把当前请求的前缀插入这个 Pod；
- 必要时淘汰缓存；
- 将本次 `miss` 加入该 Pod 的负载记录。

**4.3 主动干预**

其中有两种干预方式

**a. broadcast**

> [!important]
>
> broadcast是在请求回放开始之前进行的，将频繁出现的前缀预置到各个pod中，这样所有pod都能使用

当前实现先统计整份 Trace 中，各种“第一个 block”出现了多少次，然后选取最常见的几个，预置到所有 Pod。

例如：

```
stable_broadcast = 1
```

表示选取最常见的一种首块，放到所有 Pod。

```
stable_broadcast = 4
```

表示选取最常见的四种首块。这里的 4 表示四种候选首块，不表示复制四个 block 深的前缀。

当前 Broadcast 是直接预置缓存，代码没有为它计算传输量。

**b. oracle**

是 request_driven 的，在每条请求服务完成后进行

Oracle 可以查看未来 W 秒内的请求，用未来需求选择复制对象。阶段 2 使用：

```
W = 300 秒
cap2 = 2
depth_cap = 8 blocks
```

它的过程是：

```
查看未来 W 秒的请求
          ↓
统计各条哈希链的出现次数
          ↓
计算分数：需求次数 × 有效前缀长度
          ↓
选取得分最高的候选
          ↓
检查记录中的副本数量
          ↓
选择较空闲的目标 Pod
          ↓
检查容量与预算，插入前缀
```

这里 `depth_cap=8`，在当前 64-token 块大小下，对应最多复制 512 tokens 的前缀。



这里要特别提及holders

`holders`是oracle使用的缓存副本记录表：`holders = defaultdict(set)`

结构是`完整的请求哈希链 → Pod 编号集合`

请求服务完成或是主动干预完成后，会将前缀链记录进holders中

oracle策略选出要复制的热链之后会去查询holders，只有当持有这个热链的pod足够少（小于cap2）的时候才会启动复制，复制的target是负载最低的pod

### 模拟结果

| **模式 / 方案**               | **保存数 (saved)** | **命中率 (hit_pct)** | **倾斜度 (skew)** | **网络流量 (wire_GiB)** |
| ----------------------------- | ------------------ | -------------------- | ----------------- | ----------------------- |
| base / AFF                    | 10,984,832         | 41.9327%             | 16.5586           | 0.000000                |
| base / LEAST                  | 10,836,032         | 41.3647%             | 0.6061            | 0.000000                |
| base / REQ                    | 17,194,176         | 65.6358%             | 1.4061            | 405.111328              |
| base / REQ theta=3            | 17,062,272         | 65.1323%             | 2.3395            | 194.537109              |
| idea2 / AFF + broadcast1      | 10,984,896         | 41.9329%             | 16.5586           | 0.000000                |
| idea2 / REQ + broadcast1      | 17,194,240         | 65.6360%             | 1.4061            | 405.111328              |
| idea2 / REQ + broadcast4      | 17,194,304         | 65.6363%             | 1.4061            | 405.111328              |
| idea3 / REQ + OracleCap2 dc=8 | 17,253,120         | 65.8608%             | 0.3637            | 309.750000              |

1. AFF策略下出现了明显负载不均

2. REQ能够显著提高命中率，同时能获得较低的倾斜

3. broadcast相比于单纯的REQ提升较小（只预置了很少的block，影响较小是意料之中的）

4. oracle策略在命中率方面没有什么提升，但是请求分布更加均匀，而且主动复制减少了很多REQ的被动复制，在wire方面优化很大

   REQ本身就能够针对当前请求复制命中最高的前缀cache，所以oracle在命中率方面优化小，但是提前调度较热前缀可以明显改善请求分布，减少请求到来后的被动复制

> [!tip]
>
> 已经完成了leaf-LRU的堆优化

## stage 3

> 加存储分层，复现实验中的transfer成功率基本为0的结论

### 模拟器修改要点

1. 修改前缀树，添加host和device状态

   前缀树记录前缀路径，每个节点分别记录对应缓存是否在host和device中，并分别统计两层的容量占用

   device中的缓存被回收后，如果host中仍有备份，则前缀仍然可以用于本地复用，但是不能直接从device传输

   > 这里通过修改节点的驻留状态来实现分层，而不是分成两颗前缀树

2. 增加demote语义和写直达策略

   缓存进入device，同时将缓存备份到host。当前请求的缓存收到保护，不能demote，不被保护的缓存在容量压力下允许从device中淘汰，host本身也需要管理容量和淘汰

   整个过程可以通过修改前缀树的节点状态实现

3. 建立gateway索引和KV Event队列

   实际系统中，gateway处会维护一份缓存索引，记录各个pod持有的前缀，进行路由决策和主动干预的时候使用gateway的索引

   worker新增或删除缓存时产生event，event先进入本地队列，再周期性发布并更新gateway，因此gateway的记录可能落后于实际的缓存状态，这也是传输失败的原因之一

   在模拟器中建立gateway索引和kv event队列，还需要确定event发布的周期（暂时作为未知参数）

4. 区分pods的实际状态和gateway的索引状态

   `pods`对应实际的cache，统计真实命中、检查source能否传输时使用这个状态，缓存插入、备份、demote、淘汰和传输成功都应该修改这个状态

   `gateway`用于路由和主动干预，选择source、target以及统计前缀副本数量时使用这个索引，索引根据收到的event更新

5. 增加主动干预的调度时钟

   原本oracle是request_driven的，在每条请求服务完成后执行，根因调查文档提到gateway的主动决策有5s的tick，因此将oracle和broadcast改为周期性评估，REQ中的按需复制仍由请求触发

   不需要维护一个连续流动的时间，只需要在两个请求之间，按时间顺序处理应该发生的主动决策和event发布，每次直接跳到下一个事件的时间

6. 所有前缀复制使用同一套传输逻辑

   REQ、broadcast和oracle中的前缀复制都走统一的传输入口

   ```text
   决策器根据gateway索引选定source、target、前缀
       ↓
   根据source的真实状态检查完整前缀
       ↓
   成功：插入target，累计KV传输量，产生相应的缓存事件
   失败：记录原因和失败次数，不插入target
   ```

   当前混合模型要求source的device中有完整的待传输前缀，host直传不支持mamba，因此只有host备份时也会失败

   REQ复制失败后仍然将请求交给选定的target，由target重新计算缺失的部分

7. 增加事件记录

   | 类型 | 需要记录的内容 |
   | --- | --- |
   | 决策跳过 | 没有source、达到副本上限、目标容量不足、预算不足 |
   | 传输尝试 | 类型、时间、source、target、请求前缀长度、决策时的索引信息 |
   | 源端检查 | 实际device命中深度、host命中情况 |
   | 执行结果 | 成功、完整前缀仅在host、源端前缀已缺失 |
   | 收益与成本 | 原有命中指标、成功传输量 |

   REQ、broadcast和oracle分别统计尝试次数、成功次数、失败次数和成功率，决策跳过不计入传输失败，没有尝试时成功率显示为`--`




### 第一次测试结果

```python
python3 scripts/stage3_baseline.py --device-cap 456 --event-publish-interval-ms 1000
```





| **Case**                          | **Saved Tokens** | **Hit Rate** | **Skew** | **Wire (GiB)** | **Proactive Wire (GiB)** | **Runtime (s)** |
| --------------------------------- | ---------------- | ------------ | -------- | -------------- | ------------------------ | --------------- |
| **base AFF**                      | 10,984,832       | 41.93%       | 16.56    | 0.00           | 0.00                     | 2.64            |
| **base LEAST**                    | 10,836,032       | 41.36%       | 0.61     | 0.00           | 0.00                     | 2.69            |
| **base REQ**                      | 16,130,432       | 61.58%       | 1.35     | 30.30          | 0.00                     | 2.73            |
| **base REQ ($\theta=3$)**         | 16,674,880       | 63.65%       | 2.41     | 23.58          | 0.00                     | 2.71            |
| **idea2 AFF + broadcast1**        | 10,984,832       | 41.93%       | 16.56    | 0.00           | 0.08                     | 2.64            |
| **idea2 REQ + broadcast1**        | 16,130,496       | 61.58%       | 1.35     | 30.30          | 0.01                     | 2.62            |
| **idea2 REQ + broadcast4**        | 16,130,240       | 61.57%       | 1.35     | 30.30          | 0.22                     | 2.63            |
| **idea3 REQ + OracleCap2 (dc=8)** | 16,130,432       | 61.58%       | 1.35     | 30.30          | 0.00                     | 3.24            |

| **Case**                          | **Channel**       | **Skipped** | **Attempts** | **OK** | **Failed** | **Success Rate** | **Primary Failure Reason**                                   |
| --------------------------------- | ----------------- | ----------- | ------------ | ------ | ---------- | ---------------- | ------------------------------------------------------------ |
| **base REQ**                      | `REQ`             | 512         | 36           | 8      | 28         | 22.22%           | `host_direct_unsupported` (28)                               |
| **base REQ ($\theta=3$)**         | `REQ`             | 528         | 20           | 6      | 14         | 30.00%           | `host_direct_unsupported` (14)                               |
| **idea2 AFF + broadcast1**        | `broadcast`       | 117         | 6            | 6      | 0          | 100.00%          | None                                                         |
| **idea2 REQ + broadcast1**        | `REQ` `broadcast` | 512 118     | 36 1         | 8 1    | 28 0       | 22.22% 100.00%   | `host_direct_unsupported` (28) None                          |
| **idea2 REQ + broadcast4**        | `REQ` `broadcast` | 512 211     | 36 111       | 8 16   | 28 95      | 22.22% 14.41%    | `host_direct_unsupported` (28) `host_direct_unsupported` (95) |
| **idea3 REQ + OracleCap2 (dc=8)** | `REQ` `oracle`    | 512 119     | 36 0         | 8 0    | 28 0       | 22.22% --        | `host_direct_unsupported` (28) None                          |



- 不涉及传输的AFF和LEAST策略几乎不变

- oracle策略没有发起复制，而不是复制失败

  ```
  Oracle 候选不满足副本数或容量条件 → 没有发起复制 → 没有增量收益
  ```

  | Oracle 行为      | 旧版 | 新版 |
  | ---------------- | ---- | ---- |
  | 决策次数         | 548  | 119  |
  | 标记为成功的复制 | 26   | 0    |
  | 副本上限跳过     | 0    | 110  |
  | 目标容量跳过     | 521  | 9    |
  | 无未来需求       | 1    | 0    |

  有几个实现的问题：

  1. 旧版的副本上限判断使用的是完整的链

     ```python
     len(holders.get(ch, set())) < cap2
     ```

     所以所有的副本上限判断都可以通过。实际上副本上限判断应该判断的是要进行复制的那个前缀链

  2. 热度统计问题

     原本模拟器的实现是按照完整链来统计复用次数，实际上应该要统计的是复制部分的未来复用次数

  3. 复制的缓存行为

     设置了目标容量不足的时候跳过是否合理

  4. 目前的oracle复制只要不满足条件就会结束这次复制，是否应该进行重新选择？

- REQ大部分复制都遇到了传输失败，但是还是有少部分成功，REQ仍然有收益

- broadcast1完全成功，broadcast4成功率低，原因调查：

  1. 548 条请求只有 **两种首块**，分别出现 544 次和 4 次；后续块仍然有很多不同内容
  2. 广播每次复制的前缀长度固定为 **1 block＝64 tokens**
  3. **只要选中的 source 的 device 中存在这个首块，当前模拟器就会判定复制成功**，所以自然broadcast1成功率很高，因为它选择广播的块是大部分请求的首块，几乎一定会出现在device种

  > [!warning]
  >
  > 需要进一步确认broadcast全部失败的原因
  >
  > 所有548条请求中只有两种不同的首块，其中一种只有4个请求，大部分时间device中应该都会保留最热门的块，所以对于broadcast1，应该大部分时候都能从device中拿到要传输的块
  >
  > 按照根因调查文档中的失败机理，仅仅是因为device中没有full-kv的话不应该引起broadcast的全部失败

  > [!tip]
  >
  > 推测：模拟器中没有考虑mamba state的影响，实际sglang的前缀匹配应该会回退到具有有效mamba状态的边界，所以可能出现：
  >
  > ```
  > 前 64 tokens 的 attention KV：有
  > 处理完第 64 个 token 后的 Mamba 状态 S₆₄：没有
  > ```
  >
  > sglang中的match_prefix函数返回的长度会回退到有mamba state的边界，如果source中的检查使用match_prefix返回的长度来检查device和host是否含有要复制的前缀，就会出现错误拒绝导致广播失败，所以真实实验中出现广播全部失败的问题



### 修改oracle实现

1. 使用复制部分前缀统计热度和统计副本数
2. 即使目标满了也正常复制
3. 不满足条件时重新选取

oracle可以发生，但是仍达不到传输失败的要求，传输成功率42.55

| **Case**                          | **Channel**       | **Skipped** | **Attempts** | **OK** | **Failed** | **Success Rate**  | **Primary Failure Reason**                                   |
| --------------------------------- | ----------------- | ----------- | ------------ | ------ | ---------- | ----------------- | ------------------------------------------------------------ |
| **base REQ**                      | `REQ`             | 512         | 36           | 8      | 28         | 22.22%            | `host_direct_unsupported` (28)                               |
| **base REQ ($\theta=3$)**         | `REQ`             | 528         | 20           | 6      | 14         | 30.00%            | `host_direct_unsupported` (14)                               |
| **idea2 AFF + broadcast1**        | `broadcast`       | 117         | 6            | 6      | 0          | 100.00%           | None                                                         |
| **idea2 REQ + broadcast1**        | `REQ` `broadcast` | 512 118     | 36 1         | 8 1    | 28 0       | 22.22% 100.00%    | `host_direct_unsupported` (28) None                          |
| **idea2 REQ + broadcast4**        | `REQ` `broadcast` | 512 211     | 36 111       | 8 16   | 28 95      | 22.22% 14.41%     | `host_direct_unsupported` (28) `host_direct_unsupported` (95) |
| **idea3 REQ + OracleCap2 (dc=8)** | `REQ` `oracle`    | 517 392     | 31 47        | 7 20   | 24 27      | 22.58% **42.55%** | `host_direct_unsupported` (24) `host_direct_unsupported` (27) |

| **Case**                          | **Saved Tokens** | **Hit Rate** | **Skew** | **Wire (GiB)** | **Proactive Wire (GiB)** | **Runtime (s)** |
| --------------------------------- | ---------------- | ------------ | -------- | -------------- | ------------------------ | --------------- |
| **base AFF**                      | 10,984,832       | 41.93%       | 16.56    | 0.00           | 0.00                     | 1.61            |
| **base LEAST**                    | 10,836,032       | 41.36%       | 0.61     | 0.00           | 0.00                     | 1.74            |
| **base REQ**                      | 16,130,432       | 61.58%       | 1.35     | 30.30          | 0.00                     | 1.71            |
| **base REQ ($\theta=3$)**         | **16,674,880**   | **63.65%**   | 2.41     | 23.58          | 0.00                     | 1.73            |
| **idea2 AFF + broadcast1**        | 10,984,832       | 41.93%       | 16.56    | 0.00           | 0.08                     | 1.90            |
| **idea2 REQ + broadcast1**        | 16,130,496       | 61.58%       | 1.35     | 30.30          | 0.01                     | 1.86            |
| **idea2 REQ + broadcast4**        | 16,130,240       | 61.57%       | 1.35     | 30.30          | 0.22                     | 1.68            |
| **idea3 REQ + OracleCap2 (dc=8)** | 16,386,432       | 62.55%       | 1.85     | 27.95          | 2.19                     | 1.86            |
