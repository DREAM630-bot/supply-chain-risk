"""
Shapley 拓扑归因引擎（纯 Python，无 LLM 调用）
================================================
基于最大流特征函数与蒙特卡罗 Shapley 估计的供应链风险拓扑归因。

核心流程：
1. 从 knowledge/topology.md 同步的 TOPOLOGY 常量构建 networkx.DiGraph；
2. 特征函数 v(S) = 联盟 S 诱导子图上的 source→sink 最大流；
3. 蒙特卡罗 Shapley：N=5000 次随机排列，逐节点计算边际贡献均值；
4. 归一化到 0–1（除以总和，使 Σ responsibility = 1）；
5. main() 将结果写入 data/results.json 的 "responsibility" 字段。

本文件只包含纯 Python 代码，不包含任何 LLM 调用。
依赖仅限 requirements.txt：networkx。
"""

import json
import random
from pathlib import Path

import networkx as nx
from networkx.algorithms.flow import maximum_flow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results.json"

# ---------------------------------------------------------------------------
# 伊利液态奶 6 节点有向网络拓扑定义
# 与 knowledge/topology.md 中的 Python dict 保持同步。
# 声明：边容量为基于公开产能的相对归一化映射，
#       不声称代表企业内部真实物流货量。
# ---------------------------------------------------------------------------
TOPOLOGY = {
    "network_name": "伊利液态奶供应链6节点有向网络",
    "source": "N1",
    "sink": "N6",
    "nodes": {
        "N1": {"name": "敕勒川牧场", "role": "奶源供应商", "type": "source",
               "desc": "液态奶上游奶源供应"},
        "N2": {"name": "呼和浩特工厂", "role": "核心生产", "type": "intermediate",
               "desc": "日处理鲜奶 6500 吨"},
        "N3": {"name": "武汉分仓", "role": "区域仓储", "type": "intermediate",
               "desc": "武汉云仓日均 2 万单"},
        "N4": {"name": "冷链干线", "role": "干线物流", "type": "intermediate",
               "desc": "干线冷链运输"},
        "N5": {"name": "华南配送", "role": "末端配送", "type": "intermediate",
               "desc": "区域末端配送"},
        "N6": {"name": "零售终端", "role": "销售终端", "type": "sink",
               "desc": "终端零售销售"},
    },
    "edges": [
        {"from": "N1", "to": "N2", "capacity": 1.0},
        {"from": "N2", "to": "N3", "capacity": 0.8},
        {"from": "N3", "to": "N4", "capacity": 0.8},
        {"from": "N4", "to": "N5", "capacity": 0.6},
        {"from": "N5", "to": "N6", "capacity": 0.3},
    ],
    "disclaimer": "边容量为基于公开产能的相对归一化映射，不声称代表企业内部真实物流货量。",
}


def build_graph(topology: dict = TOPOLOGY) -> nx.DiGraph:
    """根据拓扑定义构建 networkx 有向图。"""
    G = nx.DiGraph()
    for node_id, attrs in topology["nodes"].items():
        G.add_node(node_id, **attrs)
    for edge in topology["edges"]:
        G.add_edge(edge["from"], edge["to"], capacity=edge["capacity"])
    return G


def compute_max_flow(G: nx.DiGraph, source: str, sink: str):
    """计算 source → sink 的最大流，返回 (流值, 流分布字典)。"""
    flow_value, flow_dict = maximum_flow(G, source, sink)
    return flow_value, flow_dict


def coalition_max_flow(node_set: set, topology: dict = TOPOLOGY) -> float:
    """
    特征函数 v(S)：
    仅保留联盟内节点（及两端均在联盟内的边）时的 source→sink 最大流。
    若 source 或 sink 不在联盟中，或不存在 s-t 路径，则 v(S) = 0。
    """
    source, sink = topology["source"], topology["sink"]
    if source not in node_set or sink not in node_set:
        return 0.0
    G = nx.DiGraph()
    G.add_nodes_from(node_set)
    for e in topology["edges"]:
        if e["from"] in node_set and e["to"] in node_set:
            G.add_edge(e["from"], e["to"], capacity=e["capacity"])
    if not nx.has_path(G, source, sink):
        return 0.0
    flow_value, _ = maximum_flow(G, source, sink)
    return flow_value


def precompute_coalition_values(topology: dict = TOPOLOGY) -> dict:
    """
    预计算所有 2^n 个节点子集的 v(S)，供蒙特卡罗查表加速。
    返回 {frozenset(节点集): 最大流值}。
    """
    node_ids = list(topology["nodes"].keys())
    n = len(node_ids)
    values = {}
    for mask in range(1 << n):
        S = frozenset(node_ids[i] for i in range(n) if mask & (1 << i))
        values[S] = coalition_max_flow(set(S), topology)
    return values


def monte_carlo_shapley(topology: dict = TOPOLOGY,
                        n_samples: int = 5000,
                        seed: int = 42) -> dict:
    """
    蒙特卡罗 Shapley 估计。

    对节点集合做 N 次随机排列，每次排列中逐个加入节点 i，
    计算边际贡献 v(S ∪ {i}) - v(S)，在所有排列上取平均。
    最后归一化到 0–1（除以总和，使 Σ responsibility = 1）。

    返回 {"raw": {node: 平均边际贡献},
          "normalized": {node: 归一化责任度},
          "n_samples": n_samples}
    """
    node_ids = list(topology["nodes"].keys())
    n = len(node_ids)
    rng = random.Random(seed)

    # 预计算查表，避免重复最大流求解
    lut = precompute_coalition_values(topology)

    marginal_sums = {nid: 0.0 for nid in node_ids}
    for _ in range(n_samples):
        perm = node_ids[:]
        rng.shuffle(perm)
        S = frozenset()
        v_prev = lut[S]
        for nid in perm:
            S = S | {nid}
            v_curr = lut[S]
            marginal_sums[nid] += (v_curr - v_prev)
            v_prev = v_curr

    raw = {nid: marginal_sums[nid] / n_samples for nid in node_ids}
    total = sum(raw.values())
    if total > 1e-12:
        normalized = {nid: raw[nid] / total for nid in node_ids}
    else:
        normalized = {nid: 1.0 / n for nid in node_ids}
    return {"raw": raw, "normalized": normalized, "n_samples": n_samples}


def run_attribution(topology: dict = TOPOLOGY,
                    n_samples: int = 5000) -> dict:
    """执行完整归因流程，返回可 JSON 序列化的结果。"""
    G = build_graph(topology)
    flow_value, flow_dict = compute_max_flow(
        G, topology["source"], topology["sink"]
    )
    mc = monte_carlo_shapley(topology, n_samples=n_samples)
    return {
        "network_name": topology["network_name"],
        "disclaimer": topology["disclaimer"],
        "max_flow": flow_value,
        "flow_dict": {k: v for k, v in flow_dict.items()},
        "responsibility": mc["normalized"],
        "responsibility_raw": mc["raw"],
        "n_samples": mc["n_samples"],
    }


def main():
    """计算 Shapley 责任度，写入 data/results.json 的 responsibility 字段。"""
    mc = monte_carlo_shapley(n_samples=5000)
    resp = {nid: round(v, 4) for nid, v in mc["normalized"].items()}

    # 读取已有 results.json，增量更新 responsibility
    if RESULTS_PATH.exists():
        data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        data = {"network_name": TOPOLOGY["network_name"],
                "disclaimer": TOPOLOGY["disclaimer"], "nodes": []}

    data["responsibility"] = resp
    data["n_samples"] = mc["n_samples"]

    # 若已有 nodes 结构，同步更新每个节点的 responsibility、中位数与象限
    if isinstance(data.get("nodes"), list) and data["nodes"]:
        resps = []
        for node_obj in data["nodes"]:
            nid = node_obj.get("node")
            if nid in resp:
                node_obj["responsibility"] = resp[nid]
                resps.append(resp[nid])
        if resps:
            resps_sorted = sorted(resps)
            m = len(resps_sorted)
            if m % 2 == 1:
                median_resp = resps_sorted[m // 2]
            else:
                median_resp = round(
                    (resps_sorted[m // 2 - 1] + resps_sorted[m // 2]) / 2, 4
                )
            data["median_responsibility"] = median_resp
            # 依据双中位数重新计算四象限分类，保持与 responsibility 一致
            median_exp = data.get("median_exposure")
            if median_exp is None:
                exps = sorted(n.get("exposure", 0) for n in data["nodes"])
                k = len(exps)
                median_exp = (exps[k // 2] if k % 2 == 1
                              else round((exps[k // 2 - 1] + exps[k // 2]) / 2, 4))
                data["median_exposure"] = median_exp
            for node_obj in data["nodes"]:
                exp = node_obj.get("exposure", 0)
                r = node_obj.get("responsibility", 0)
                if exp >= median_exp and r >= median_resp:
                    node_obj["quadrant"] = "显性核心"
                elif exp < median_exp and r >= median_resp:
                    node_obj["quadrant"] = "隐性关键"
                elif exp >= median_exp and r < median_resp:
                    node_obj["quadrant"] = "局部扰动"
                else:
                    node_obj["quadrant"] = "常规节点"

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[shapley_engine] 责任度已写入 {RESULTS_PATH}")
    print(f"[shapley_engine] 归一化责任度: {resp}")
    print(f"[shapley_engine] 蒙特卡罗样本数: {mc['n_samples']}")


if __name__ == "__main__":
    main()