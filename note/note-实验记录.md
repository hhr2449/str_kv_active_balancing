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