"""多 Pod KV Cache 回放模拟器 v3：同一棵前缀树上的 Device/Host 分层缓存。

口径对齐学生 T2 补充实验：块=512 token=1 page；前缀树 + leaf-LRU；负载=最近60s miss tokens；
命中 tokens = min(hit_depth*512, input_length)；wire 按 14 MiB/page；评价窗 [1500,2700)s。
本轮仅增加分层驻留、写直达和容量回收；请求仍按瞬时完成处理。
"""

import json, math, random, sys
import heapq
from collections import defaultdict, deque
from bisect import bisect_right
from dataclasses import dataclass

BLOCK = 512
PAGE_BYTES = 14 * 1024 * 1024
GIB = 2 ** 30


def load_trace(path):
    reqs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            reqs.append((int(r["timestamp"]), int(r["input_length"]),
                         int(r["output_length"]), tuple(r["hash_ids"])))
    reqs.sort(key=lambda x: x[0])
    return reqs


@dataclass
class _CacheNode:
    # 一棵树记录前缀路径，两层分别记录缓存是否驻留。
    parent: int
    block_hash: object
    last_access: int
    in_device: bool = False
    in_host: bool = False


class PodCache:
    def __init__(self, capacity_pages, infinite=False, device_cap=None):
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
            if not node.in_device:
                node.in_device = True
                self.device_size += 1
            if not node.in_host:
                node.in_host = True
                self.host_size += 1
            parent = nid
        # 请求按瞬时完成处理，写入后再回收，不设置执行期保护。
        self._evict()

    def _evict(self):
        if self.infinite:
            return
        self._evict_layer("device")
        self._evict_layer("host")

    def _evict_layer(self, layer):
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
        scan_limit=1500, device_cap=None):
    # C 为每个 Pod 的 Host 容量，device_cap 单独控制 Device 容量（单位均为块）。
    pods = [PodCache(C, infinite=infinite, device_cap=device_cap) for _ in range(N)]
    loads = [deque() for _ in range(N)]

    def pod_load(i, t):
        s = 0
        for (ts, m) in loads[i]:
            if ts > t - 60000:
                s += m
        return s

    # idea2: 稳定前缀广播（最常出现的首块，预置到所有实例）
    if stable_broadcast > 0:
        freq = defaultdict(int)
        for (t, il, ol, h) in reqs:
            freq[h[0]] += 1
        top = sorted(freq, key=freq.get, reverse=True)[:stable_broadcast]
        for p in pods:
            for b in top:
                p.insert((b,), 0)

    saved_eval = 0
    input_eval = 0
    wire_eval = 0
    proactive_wire_eval = 0
    proactive_wire_full = 0
    pod_req_eval = [0] * N
    holders = defaultdict(set)   # 整链身份 -> 已发布副本的实例集合

    for idx, (t, il, ol, h) in enumerate(reqs):
        # 1) routing（AFF/LEAST/REQ 共享，主动策略只是叠加，不改路由）
        if routing == "AFF":
            src = -1
            bd = -1
            bl = None
            for p in range(N):
                d = pods[p].hit_depth(h, t)
                if d > bd:
                    src, bd, bl = p, d, pod_load(p, t)
                elif d == bd:
                    l = pod_load(p, t)
                    if l < bl:
                        src, bd, bl = p, d, l
        elif routing == "LEAST":
            src = min(range(N), key=lambda p: (pod_load(p, t), p))
        elif routing == "REQ":
            src = -1
            bd = -1
            bl = None
            for p in range(N):
                d = pods[p].hit_depth(h, t)
                if d > bd:
                    src, bd, bl = p, d, pod_load(p, t)
                elif d == bd:
                    l = pod_load(p, t)
                    if l < bl:
                        src, bd, bl = p, d, l
            if bl is not None:
                others = [pod_load(p, t) for p in range(N) if p != src]
                if others and bl > theta * min(others):
                    target = min(range(N), key=lambda p: (pod_load(p, t), p))
                    prefix = h[:bd]
                    if prefix and target != src and not pods[target].contains(prefix):
                        pods[target].insert(prefix, t)
                        if eval_start <= t < eval_end:
                            wire_eval += len(prefix) * PAGE_BYTES
                    src = target

        # 2) serve
        d = pods[src].hit_depth(h, t)
        saved = min(d * BLOCK, il)
        miss = il - saved
        if eval_start <= t < eval_end:
            saved_eval += saved
            input_eval += il
            pod_req_eval[src] += 1
        pods[src].insert(h, t)
        loads[src].append((t, miss))
        holders[h].add(src)

        # 3) proactive（Oracle 类：未来 W 秒需求排序 + 副本上限 + 深度限制 + 预算）
        if policy == "Oracle":
            demand = defaultdict(int)
            j = idx + 1
            while j < len(reqs) and reqs[j][0] <= t + W and j - idx <= scan_limit:
                demand[reqs[j][3]] += 1
                j += 1
            if demand:
                best_chain = None
                best_score = -1
                for ch, dd in demand.items():
                    eff = min(len(ch), depth_cap) if depth_cap else len(ch)
                    score = dd * eff if demand_weight_depth else dd
                    if score > best_score:
                        best_score = score
                        best_chain = ch
                ch = best_chain
                if len(holders.get(ch, set())) < cap2:
                    eff = min(len(ch), depth_cap) if depth_cap else len(ch)
                    target = None
                    bl = None
                    for p in range(N):
                        if p in holders.get(ch, set()):
                            continue
                        l = pod_load(p, t)
                        if bl is None or l < bl:
                            bl = l
                            target = p
                    need = eff * PAGE_BYTES
                    if target is not None and (pods[target].infinite or pods[target].size + eff <= pods[target].cap):
                        if budget_bytes is None or proactive_wire_full + need <= budget_bytes:
                            pods[target].insert(ch[:eff], t)
                            holders[ch].add(target)
                            proactive_wire_full += need
                            if eval_start <= t < eval_end:
                                proactive_wire_eval += need

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
