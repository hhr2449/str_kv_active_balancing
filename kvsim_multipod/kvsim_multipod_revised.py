"""多 Pod KV Cache 回放模拟器 v3：同一棵前缀树上的 Device/Host 分层缓存。

口径对齐学生 T2 补充实验：块=512 token=1 page；前缀树 + leaf-LRU；负载=最近60s miss tokens；
命中 tokens = min(hit_depth*512, input_length)；wire 按 14 MiB/page；评价窗 [1500,2700)s。
缓存事件周期发布到 gateway，主动策略每 5 秒评估；请求仍按瞬时完成处理。
"""

import json, math, random, sys
import heapq
from collections import defaultdict, deque
from bisect import bisect_right
from dataclasses import dataclass

BLOCK = 512
PAGE_BYTES = 14 * 1024 * 1024
GIB = 2 ** 30
PROACTIVE_INTERVAL_MS = 5000


def load_trace(path):
    reqs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            reqs.append((int(r["timestamp"]), int(r["input_length"]),
                         int(r["output_length"]), tuple(r["hash_ids"])))
    reqs.sort(key=lambda x: x[0])
    return reqs


@dataclass(frozen=True)
class _KVEvent:
    # 保存变更当时的驻留快照，发布时不再读取 worker 的最新状态。
    timestamp: int
    nid: int
    parent: int
    block_hash: object
    in_device: bool
    in_host: bool


@dataclass
class _CacheNode:
    # 一棵树记录前缀路径，两层分别记录缓存是否驻留。
    parent: int
    block_hash: object
    last_access: int
    in_device: bool = False
    in_host: bool = False


class PodCache:
    def __init__(self, capacity_pages, infinite=False, device_cap=None, event_queue=None):
        if capacity_pages < 0 or (device_cap is not None and device_cap < 0):
            raise ValueError("缓存容量不能为负数")
        # 原容量参数表示 Host 容量；Device 未单独指定时使用相同容量。
        self.host_cap = capacity_pages if not infinite else None
        self.device_cap = (capacity_pages if device_cap is None else device_cap) if not infinite else None
        self.infinite = infinite
        self.nodes = {}
        self.children = defaultdict(dict)
        self.roots = {}
        self.device_size = 0
        self.host_size = 0
        self.next_id = 0
        # 回放时接入 worker 本地队列；天花板计算不需要记录事件。
        self.kv_events = event_queue

    def _emit_event(self, nid, t):
        if self.kv_events is not None:
            node = self.nodes[nid]
            self.kv_events.append(_KVEvent(t, nid, node.parent, node.block_hash,
                                           node.in_device, node.in_host))

    # 兼容原策略的容量检查，cap/size 均表示 Host 层。
    @property
    def cap(self):
        return self.host_cap

    @property
    def size(self):
        return self.host_size

    def _match_depth(self, blocks, layer, t=None):
        if layer not in ("local", "device", "host"):
            raise ValueError("layer 必须是 local、device 或 host")
        parent = -1
        depth = 0
        for b in blocks:
            nid = self.children.get(parent, {}).get(b) if parent != -1 else self.roots.get(b)
            if nid is None:
                break
            node = self.nodes[nid]
            if layer == "local":
                resident = node.in_device or node.in_host
            elif layer == "device":
                resident = node.in_device
            else:
                resident = node.in_host
            # 路径存在不等于缓存命中，任一块缺失就停止连续匹配。
            if not resident:
                break
            if t is not None:
                node.last_access = t
            parent = nid
            depth += 1
        return depth

    def hit_depth(self, blocks, t, layer="local"):
        # 本地逐块复用两层缓存；指定 layer 可查询单层的连续驻留深度。
        return self._match_depth(blocks, layer, t)

    def insert(self, blocks, t):
        parent = -1
        for b in blocks:
            nid = self.children.get(parent, {}).get(b) if parent != -1 else self.roots.get(b)
            if nid is None:
                self.next_id += 1
                nid = self.next_id
                self.nodes[nid] = _CacheNode(parent, b, t)
                if parent == -1:
                    self.roots[b] = nid
                else:
                    self.children[parent][b] = nid
            node = self.nodes[nid]
            node.last_access = t
            # 写入 Device，同时备份到 Host；已有副本不重复计容量。
            changed = False
            if not node.in_device:
                node.in_device = True
                self.device_size += 1
                changed = True
            if not node.in_host:
                node.in_host = True
                self.host_size += 1
                changed = True
            if changed:
                self._emit_event(nid, t)
            parent = nid
        # 请求按瞬时完成处理，写入后再回收，不设置执行期保护。
        self._evict(t)

    def _evict(self, t):
        if self.infinite:
            return
        self._evict_layer("device", t)
        self._evict_layer("host", t)

    def _evict_layer(self, layer, t):
        cap = getattr(self, f"{layer}_cap")
        size_attr = f"{layer}_size"
        resident_attr = f"in_{layer}"
        if cap is None or getattr(self, size_attr) <= cap:
            return

        # 两层各自做 leaf-LRU，另一层的子节点不会阻止本层回收。
        resident_children = defaultdict(int)
        for node in self.nodes.values():
            if getattr(node, resident_attr):
                resident_children[node.parent] += 1
        # 同时间戳按递增 nid 选择，沿用原来的平手规则。
        leaves = [(node.last_access, nid) for nid, node in self.nodes.items()
                  if getattr(node, resident_attr) and not resident_children[nid]]
        heapq.heapify(leaves)
        while getattr(self, size_attr) > cap:
            _, best = heapq.heappop(leaves)
            node = self.nodes[best]
            # Device 回收即 demote，只清除 Device 状态，保留已有 Host 备份。
            setattr(node, resident_attr, False)
            setattr(self, size_attr, getattr(self, size_attr) - 1)
            # 删除树节点之前记录事件，保证发布时仍有节点身份和父路径。
            self._emit_event(best, t)
            par = node.parent
            resident_children[par] -= 1
            if par != -1 and not resident_children[par]:
                parent_node = self.nodes[par]
                if getattr(parent_node, resident_attr):
                    heapq.heappush(leaves, (parent_node.last_access, par))
            self._prune_empty(best)

    def _prune_empty(self, nid):
        # 两层都没有副本且没有子节点时，才删除树上的路径节点。
        while nid != -1:
            node = self.nodes[nid]
            if node.in_device or node.in_host or self.children.get(nid):
                return
            del self.nodes[nid]
            self.children.pop(nid, None)
            if node.parent == -1:
                del self.roots[node.block_hash]
            else:
                del self.children[node.parent][node.block_hash]
            nid = node.parent

    def contains(self, blocks, layer="local"):
        # 只检查连续驻留，不刷新 LRU 时间。
        blocks = tuple(blocks)
        return self._match_depth(blocks, layer) == len(blocks)


class GatewayIndex:
    def __init__(self, num_pods):
        # 索引按 Pod 保存前缀和驻留状态，不参与真实缓存的 LRU。
        self.nodes = [{} for _ in range(num_pods)]
        self.children = [defaultdict(dict) for _ in range(num_pods)]
        self.roots = [{} for _ in range(num_pods)]
        self.device_sizes = [0] * num_pods
        self.host_sizes = [0] * num_pods

    def apply(self, pod, event):
        nodes = self.nodes[pod]
        old = nodes.get(event.nid)
        self.device_sizes[pod] += int(event.in_device) - int(old.in_device if old else False)
        self.host_sizes[pod] += int(event.in_host) - int(old.in_host if old else False)
        nodes[event.nid] = event
        if event.parent == -1:
            self.roots[pod][event.block_hash] = event.nid
        else:
            self.children[pod][event.parent][event.block_hash] = event.nid
        # 持有关系看两层并集，Device 回收且 Host 仍在时不会删除前缀。
        if not event.in_device and not event.in_host:
            self._prune_empty(pod, event.nid)

    def _prune_empty(self, pod, nid):
        nodes = self.nodes[pod]
        children = self.children[pod]
        while nid != -1:
            node = nodes[nid]
            if node.in_device or node.in_host or children.get(nid):
                return
            del nodes[nid]
            children.pop(nid, None)
            if node.parent == -1:
                del self.roots[pod][node.block_hash]
            else:
                del children[node.parent][node.block_hash]
            nid = node.parent

    def hit_depth(self, pod, blocks, layer="local"):
        if layer not in ("local", "device", "host"):
            raise ValueError("layer 必须是 local、device 或 host")
        parent = -1
        depth = 0
        for b in blocks:
            nid = (self.roots[pod].get(b) if parent == -1
                   else self.children[pod].get(parent, {}).get(b))
            if nid is None:
                break
            node = self.nodes[pod][nid]
            if layer == "local":
                resident = node.in_device or node.in_host
            elif layer == "device":
                resident = node.in_device
            else:
                resident = node.in_host
            if not resident:
                break
            parent = nid
            depth += 1
        return depth

    def contains(self, pod, blocks, layer="local"):
        blocks = tuple(blocks)
        return self.hit_depth(pod, blocks, layer) == len(blocks)

    def holders(self, blocks):
        # 副本数量也只看已发布索引，按实际要复制的前缀查询。
        blocks = tuple(blocks)
        return {p for p in range(len(self.nodes)) if self.contains(p, blocks)}


def gini(xs):
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    s = 0.0
    for a in xs:
        for b in xs:
            s += abs(a - b)
    return s / (2 * n * sum(xs))


def random_gini(N, total, samples=200, seed=20260911):
    rng = random.Random(seed)
    gs = []
    for _ in range(samples):
        counts = [0] * N
        for _ in range(total):
            counts[rng.randrange(N)] += 1
        gs.append(gini(counts))
    return sum(gs) / len(gs)


def potential_ceiling(reqs, eval_start=1500000, eval_end=2700000):
    c = PodCache(0, infinite=True)
    saved = 0
    input_tot = 0
    for (t, il, ol, h) in reqs:
        d = c.hit_depth(h, t)
        s = min(d * BLOCK, il)
        if eval_start <= t < eval_end:
            saved += s
            input_tot += il
        c.insert(h, t)
    return saved, input_tot


def run(reqs, N, C, routing="AFF", policy=None, theta=2.0, W=300000, infinite=False,
        eval_start=1500000, eval_end=2700000, cap2=2, depth_cap=None,
        budget_bytes=None, stable_broadcast=0, demand_weight_depth=True,
        scan_limit=1500, device_cap=None, event_publish_interval_ms=1000):
    # 发布周期独立配置，默认 1 秒；主动评估固定为 5 秒。
    if event_publish_interval_ms <= 0:
        raise ValueError("事件发布周期必须大于 0 毫秒")
    # 使用 Trace 毫秒时间；稳定排序保留同一时间戳的请求顺序。
    reqs = sorted(reqs, key=lambda r: r[0])
    # C 为每个 Pod 的 Host 容量，device_cap 单独控制 Device 容量（单位均为块）。
    kv_event_queues = [deque() for _ in range(N)]
    pods = [PodCache(C, infinite=infinite, device_cap=device_cap,
                     event_queue=kv_event_queues[p]) for p in range(N)]
    gateway = GatewayIndex(N)
    loads = [deque() for _ in range(N)]

    def pod_load(i, t):
        s = 0
        for (ts, m) in loads[i]:
            if ts > t - 60000:
                s += m
        return s

    # 保留按整份 Trace 选择热门首块的口径，改由周期时钟触发广播。
    broadcast_top = []
    if stable_broadcast > 0:
        freq = defaultdict(int)
        for (t, il, ol, h) in reqs:
            if h:
                freq[h[0]] += 1
        broadcast_top = sorted(freq, key=freq.get, reverse=True)[:stable_broadcast]

    saved_eval = 0
    input_eval = 0
    wire_eval = 0
    proactive_wire_eval = 0
    proactive_wire_full = 0
    pod_req_eval = [0] * N

    def publish_events(t):
        # 只有发布时才更新 gateway，worker 的真实状态不会直接写入索引。
        for p, queue in enumerate(kv_event_queues):
            while queue and queue[0].timestamp <= t:
                gateway.apply(p, queue.popleft())

    def copy_prefix(source, target, prefix, t, proactive=False):
        nonlocal wire_eval, proactive_wire_eval, proactive_wire_full
        if source == target or not prefix:
            return False
        need = len(prefix) * PAGE_BYTES
        # 周期 Broadcast 和 Oracle 共用主动流量与预算口径。
        if proactive and budget_bytes is not None and proactive_wire_full + need > budget_bytes:
            return False
        # 本轮仍直接写入目标，源端 Device 校验后续接入；新副本也需等待事件发布。
        pods[target].insert(prefix, t)
        if proactive:
            proactive_wire_full += need
            if eval_start <= t < eval_end:
                proactive_wire_eval += need
        elif eval_start <= t < eval_end:
            wire_eval += need
        return True

    def proactive_tick(t, next_request):
        # Broadcast 从索引中的持有者选源，不再在回放前凭空预置缓存。
        for b in broadcast_top:
            prefix = (b,)
            holders = gateway.holders(prefix)
            if not holders:
                continue
            source = min(holders, key=lambda p: (pod_load(p, t), p))
            for target in range(N):
                if target not in holders:
                    copy_prefix(source, target, prefix, t, proactive=True)

        if policy != "Oracle":
            return
        # 每个 tick 重新查看 [t, t+W] 内尚未处理的请求，而非沿用上一请求时间。
        demand = defaultdict(int)
        j = next_request
        while j < len(reqs) and reqs[j][0] <= t + W and j - next_request < scan_limit:
            if reqs[j][3]:
                demand[reqs[j][3]] += 1
            j += 1
        if not demand:
            return
        best_chain = None
        best_score = -1
        for ch, dd in demand.items():
            eff = min(len(ch), depth_cap) if depth_cap else len(ch)
            score = dd * eff if demand_weight_depth else dd
            if score > best_score:
                best_score = score
                best_chain = ch
        eff = min(len(best_chain), depth_cap) if depth_cap else len(best_chain)
        prefix = best_chain[:eff]
        holders = gateway.holders(prefix)
        if not prefix or not holders or len(holders) >= cap2:
            return
        source = min(holders, key=lambda p: (pod_load(p, t), p))
        targets = [p for p in range(N) if p not in holders]
        if not targets:
            return
        target = min(targets, key=lambda p: (pod_load(p, t), p))
        # 容量预估也只使用已发布索引，已有 Host 副本不重复占用空间。
        extra_host = len(prefix) - gateway.hit_depth(target, prefix, layer="host")
        if not infinite and gateway.host_sizes[target] + extra_host > C:
            return
        copy_prefix(source, target, prefix, t, proactive=True)

    # 两个周期都从 Trace 首个请求开始计时，不依赖宿主机时间。
    first_t = reqs[0][0] if reqs else 0
    next_publish = first_t + event_publish_interval_ms
    next_proactive = (first_t + PROACTIVE_INTERVAL_MS
                      if broadcast_top or policy == "Oracle" else math.inf)

    for idx, (t, il, ol, h) in enumerate(reqs):
        # 直接跳到下一事件；同刻顺序为发布、主动决策、请求，同刻请求保持原序。
        while min(next_publish, next_proactive) <= t:
            if next_publish <= next_proactive:
                publish_events(next_publish)
                next_publish += event_publish_interval_ms
            else:
                proactive_tick(next_proactive, idx)
                next_proactive += PROACTIVE_INTERVAL_MS

        # 路由只查询 gateway，不刷新任何 worker 缓存的 LRU。
        if routing in ("AFF", "REQ"):
            depths = [gateway.hit_depth(p, h) for p in range(N)]
            src = min(range(N), key=lambda p: (-depths[p], pod_load(p, t), p))
            if routing == "REQ":
                bl = pod_load(src, t)
                others = [pod_load(p, t) for p in range(N) if p != src]
                if others and bl > theta * min(others):
                    target = min(range(N), key=lambda p: (pod_load(p, t), p))
                    prefix = h[:depths[src]]
                    if prefix and target != src and not gateway.contains(target, prefix):
                        # REQ 仍由当前请求触发，不等待下一个主动决策 tick。
                        copy_prefix(src, target, prefix, t)
                    src = target
        elif routing == "LEAST":
            src = min(range(N), key=lambda p: (pod_load(p, t), p))
        else:
            raise ValueError("routing 必须是 AFF、LEAST 或 REQ")

        # 实际命中始终查询 pods，索引滞后不会被当作真实命中。
        d = pods[src].hit_depth(h, t)
        saved = min(d * BLOCK, il)
        miss = il - saved
        if eval_start <= t < eval_end:
            saved_eval += saved
            input_eval += il
            pod_req_eval[src] += 1
        pods[src].insert(h, t)
        loads[src].append((t, miss))

    token_hit = saved_eval / input_eval if input_eval else 0.0
    skew = gini(pod_req_eval) / random_gini(N, sum(pod_req_eval)) if sum(pod_req_eval) else 0.0
    return dict(saved=saved_eval, input=input_eval, token_hit=token_hit, skew=skew,
                wire_GiB=wire_eval / GIB, proactive_wire_GiB=proactive_wire_eval / GIB,
                proactive_wire_full_GiB=proactive_wire_full / GIB,
                pod_req=pod_req_eval)


if __name__ == "__main__":
    wl = sys.argv[1] if len(sys.argv) > 1 else "conversation"
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    path = f"data/mooncake/{wl}_trace.jsonl"
    reqs = load_trace(path)
    N, C = 16, 585
    pc_saved, pc_input = potential_ceiling(reqs)
    print(f"# {wl}  ceiling={pc_saved/1e6:.3f}M")

    def show(tag, **kw):
        r = run(reqs, N, C, **kw)
        sw = (f"{r['saved']/(1e6*(r['proactive_wire_GiB'])):6.2f}" if r['proactive_wire_GiB'] > 0 else "  --  ")
        print(f"{tag:38s} saved={r['saved']/1e6:7.3f}M hit={r['token_hit']*100:5.2f}% "
              f"skew={r['skew']:6.2f} pwire={r['proactive_wire_GiB']:8.2f}GiB "
              f"pwirefull={r['proactive_wire_full_GiB']:9.2f}GiB saved/GBwire={sw}")

    if mode in ("base", "all"):
        show("AFF", routing="AFF")
        show("LEAST", routing="LEAST")
        show("REQ", routing="REQ")
        show("REQ theta=3", routing="REQ", theta=3.0)
    if mode in ("idea1", "all"):
        for dc in [None, 32, 16, 8, 4, 2]:
            show(f"OracleCap2(AFF) depth_cap={dc}", routing="AFF", policy="Oracle", cap2=2, depth_cap=dc)
    if mode in ("idea2", "all"):
        show("AFF + broadcast1", routing="AFF", stable_broadcast=1)
        show("REQ + broadcast1", routing="REQ", stable_broadcast=1)
        show("REQ + broadcast4", routing="REQ", stable_broadcast=4)
    if mode in ("idea3", "all"):
        show("REQ + OracleCap2 dc=8", routing="REQ", policy="Oracle", cap2=2, depth_cap=8)
    if mode in ("idea4", "all"):
        no_budget = run(reqs, N, C, routing="REQ", policy="Oracle", cap2=2, depth_cap=8)
        full = no_budget["proactive_wire_full_GiB"] * GIB
        for frac in [0.5, 0.25, 0.1]:
            show(f"REQ+OracleCap2 dc=8 budget={int(frac*100)}%", routing="REQ", policy="Oracle",
                 cap2=2, depth_cap=8, budget_bytes=int(full * frac))
    if mode in ("idea5", "all"):
        show("REQ+OracleCap2 dc=8 value=demand*depth", routing="REQ", policy="Oracle", cap2=2, depth_cap=8, demand_weight_depth=True)
        show("REQ+OracleCap2 dc=8 value=demand", routing="REQ", policy="Oracle", cap2=2, depth_cap=8, demand_weight_depth=False)
