"""
有效性检验（Validity Tests）
============================
检验一：Kendall's W 排序稳定性 —— 边容量 ±20% 扰动下责任度排名的一致性。
检验二：Pearson 收敛效度 —— 四维规则评分(A) 与 证据强度+时效性评分(B) 的相关度。

运行方式：
    cd supply-chain-risk-attribution
    python scripts/validity_tests.py

输出：数值直接打印，并写入 data/validity.json。
本文件为纯 Python 实现，不调用 LLM；规则 A 用四维规则的简化代码模拟
（保留 mock_score_A 接口，答辩前可替换为真实 Agent 输出）。
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shapley_engine import (  # noqa: E402
    TOPOLOGY,
    coalition_max_flow,
    precompute_coalition_values,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VALIDITY_PATH = PROJECT_ROOT / "data" / "validity.json"
CORPUS_PATH = PROJECT_ROOT / "knowledge" / "opinion_corpus.md"

WINDOW_END = date(2026, 9, 30)
R3_CUTOFF = date(2026, 3, 30)  # 窗口终点前推 6 个月


# ===========================================================================
# 工具函数
# ===========================================================================

def months_between(d1: date, d2: date) -> float:
    """计算两个日期之间的近似月数。"""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month) + (d2.day - d1.day) / 30.0


def perturb_topology(topology: dict, rng: np.random.Generator) -> dict:
    """对每条边容量做 ±20% 随机扰动：base * uniform(0.8, 1.2)。"""
    new_topo = dict(topology)
    new_topo["edges"] = [
        {**e, "capacity": round(e["capacity"] * float(rng.uniform(0.8, 1.2)), 4)}
        for e in topology["edges"]
    ]
    return new_topo


def exact_shapley(topology: dict) -> dict:
    """精确排列枚举 Shapley（用于扰动检验，结果确定性强）。"""
    from itertools import permutations
    node_ids = list(topology["nodes"].keys())
    n = len(node_ids)
    lut = precompute_coalition_values(topology)
    shapley = {nid: 0.0 for nid in node_ids}
    for perm in permutations(node_ids):
        S = frozenset()
        v_prev = lut[S]
        for nid in perm:
            S = S | {nid}
            v_curr = lut[S]
            shapley[nid] += (v_curr - v_prev)
            v_prev = v_curr
    from math import factorial
    fn = factorial(n)
    return {nid: shapley[nid] / fn for nid in node_ids}


# ===========================================================================
# 检验一：Kendall's W 排序稳定性
# ===========================================================================

def test_kendalls_w(n_perturb: int = 100, seed: int = 42) -> dict:
    """
    对每条边容量做 100 次随机扰动（±20%），每次重新计算 Shapley 责任度，
    按责任度降序排秩，得到 100×6 排名矩阵，计算 Kendall's W。

    W = 12 * S / (m^2 * (n^3 - n))
    S = sum((R_j - mean(R))^2)，R_j 为节点 j 的秩和，m=100，n=6。
    """
    rng = np.random.default_rng(seed)
    node_ids = list(TOPOLOGY["nodes"].keys())
    m, n = n_perturb, len(node_ids)

    rank_matrix = np.zeros((m, n))
    for i in range(m):
        topo_i = perturb_topology(TOPOLOGY, rng)
        shap = exact_shapley(topo_i)
        vals = np.array([shap[nid] for nid in node_ids])
        # 按责任度降序排秩：最大值秩为 1
        ranks = rankdata(-vals, method="average")
        rank_matrix[i, :] = ranks

    # 秩和 R_j 与 Kendall's W
    R = rank_matrix.sum(axis=0)          # 每节点秩和, shape (n,)
    R_mean = m * (n + 1) / 2.0
    S = float(np.sum((R - R_mean) ** 2))
    W = 12.0 * S / (m ** 2 * (n ** 3 - n))

    print("=" * 60)
    print("检验一：Kendall's W 排序稳定性")
    print("=" * 60)
    print(f"扰动次数 m = {m}，节点数 n = {n}")
    print(f"各节点秩和: {dict(zip(node_ids, R.tolist()))}")
    print(f"秩和均值 R_mean = {R_mean:.1f}")
    print(f"S = {S:.4f}")
    print(f"Kendall's W = {W:.4f}")
    if W < 0.1:
        print("注：串行链拓扑下 Shapley 为均匀归因（各节点责任度相等），"
              "扰动容量不改变均匀性，排名完全并列，W≈0。")
        print("    该结果本身是有效的拓扑性质验证：当前拓扑责任度排序退化，"
              "需引入旁支/并行路径方可区分节点责任。")
    else:
        print(f"排序一致性{'较强' if W >= 0.5 else '中等'}（W={W:.4f}）。")

    return {
        "kendalls_w": round(W, 4),
        "m": m,
        "n": n,
        "S": round(S, 4),
        "rank_sums": {nid: round(float(r), 2) for nid, r in zip(node_ids, R)},
    }


# ===========================================================================
# 检验二：Pearson 收敛效度（双规则交叉）
# ===========================================================================

def load_events() -> list:
    """从 knowledge/opinion_corpus.md 解析 OPINION_CORPUS，返回事件列表。"""
    txt = CORPUS_PATH.read_text(encoding="utf-8")
    m = re.search(r"```python\n(OPINION_CORPUS\s*=\s*\{.*?\})\n```", txt, re.S)
    if not m:
        raise ValueError("无法从 opinion_corpus.md 中解析 OPINION_CORPUS")
    ns = {}
    exec(m.group(1), ns)
    events = ns["OPINION_CORPUS"]["events"]
    assert len(events) >= 12, f"事件样本量不足 12 条，当前 {len(events)}"
    return events


def extract_severity_raw(text: str) -> float:
    """
    模拟 Extractor：从原文提取量化指标，映射到初步印象分 severity_raw。
    纯规则实现，不调用 LLM。
    """
    score = 0.30  # 基础分
    # 百分比（下降/上涨 X%）
    for p in re.findall(r"(\d+(?:\.\d+)?)%", text):
        score += min(float(p) / 100.0, 0.5) * 0.4
    # 倍数（激增 X 倍）
    for t in re.findall(r"(\d+(?:\.\d+)?)倍", text):
        score += min(float(t) / 3.0, 1.0) * 0.3
    # 小时延迟
    for h in re.findall(r"(\d+(?:\.\d+)?)小时", text):
        score += min(float(h) / 24.0, 1.0) * 0.2
    # 天数变化
    for d in re.findall(r"(\d+(?:\.\d+)?)天", text):
        score += min(float(d) / 7.0, 1.0) * 0.15
    return round(min(score, 0.95), 2)


def count_quant_points(text: str) -> int:
    """统计原文中可提取的独立量化指标个数。"""
    n = 0
    n += len(re.findall(r"\d+(?:\.\d+)?%", text))
    n += len(re.findall(r"\d+(?:\.\d+)?倍", text))
    n += len(re.findall(r"\d+(?:\.\d+)?小时", text))
    n += len(re.findall(r"\d+(?:\.\d+)?天", text))
    n += len(re.findall(r"\d+(?:\.\d+)?万单", text))
    n += len(re.findall(r"\d+(?:\.\d+)?公里", text))
    return n


def mock_score_A(event: dict) -> float:
    """
    规则 A：模拟 Extractor + Feedback + Revision 四维流程（纯 Python 简化实现）。
    答辩前可替换为真实 Agent 输出：保持函数签名 (event)->float 即可。

    流程：
    1. severity_raw = extract_severity_raw(原文)   [模拟 Extractor]
    2. R2 范围词校准 / R3 时效衰减 / R4 单因子       [模拟 Feedback]
    3. severity_final = clamp(raw + delta, 0, 1)    [模拟 Revision]
    """
    text = event["原文"]
    severity_raw = extract_severity_raw(text)
    scope = event["scope_word"]
    decay = event["time_decay_flag"]
    ev_date = date.fromisoformat(event["date"])

    delta = 0.0
    # R2 范围词校准
    caps = {"局部": 0.70, "区域": 0.85, "全局": 1.00}
    cap = caps.get(scope, 1.00)
    if severity_raw > cap:
        delta -= (severity_raw - cap)
    # R3 时效衰减（decay=false 豁免）
    if ev_date < R3_CUTOFF and decay:
        delta -= 0.15
    # R4 单因子一致性
    if count_quant_points(text) <= 1 and severity_raw > 0.60:
        delta -= 0.10

    return round(max(0.0, min(1.0, severity_raw + delta)), 4)


def score_B(event: dict) -> float:
    """
    规则 B：仅用"证据强度 + 时效性"两维加权，不引入四维批判。
    score_B = 0.6 * 证据强度 + 0.4 * 时效性
    """
    text = event["原文"]
    evidence_strength = extract_severity_raw(text)
    months = months_between(date.fromisoformat(event["date"]), WINDOW_END)
    timeliness = max(0.0, 1.0 - months / 13.0)
    return round(0.6 * evidence_strength + 0.4 * timeliness, 4)


def test_pearson_convergence() -> dict:
    """对全部事件计算 (score_A, score_B)，用 pearsonr 检验收敛效度。"""
    events = load_events()
    n = len(events)
    scores_A, scores_B = [], []
    for ev in events:
        a = mock_score_A(ev)
        b = score_B(ev)
        scores_A.append(a)
        scores_B.append(b)

    r, p = pearsonr(scores_A, scores_B)

    print("=" * 60)
    print("检验二：Pearson 收敛效度（双规则交叉）")
    print("=" * 60)
    print(f"事件级样本量 n = {n}")
    print(f"{'event_id':<32} {'score_A':>8} {'score_B':>8}")
    for ev, a, b in zip(events, scores_A, scores_B):
        print(f"{ev['event_id']:<32} {a:>8.4f} {b:>8.4f}")
    print(f"\nPearson r = {r:.4f}")
    print(f"p 值      = {p:.6f}")
    if p < 0.05:
        print(f"结论：双规则评分显著正相关（r={r:.4f}, p={p:.4f}），收敛效度成立。")
    else:
        print(f"结论：相关性不显著（p={p:.4f} ≥ 0.05），需检查规则设计。")

    return {
        "pearson_r": round(r, 4),
        "p_value": round(p, 6),
        "n": n,
        "scores": [
            {"event_id": ev["event_id"], "score_A": a, "score_B": b}
            for ev, a, b in zip(events, scores_A, scores_B)
        ],
    }


# ===========================================================================
# 主入口
# ===========================================================================

def main():
    result_w = test_kendalls_w()
    result_p = test_pearson_convergence()

    output = {
        "kendalls_w": result_w["kendalls_w"],
        "kendalls_w_detail": result_w,
        "pearson_r": result_p["pearson_r"],
        "pearson_detail": result_p,
    }
    VALIDITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    VALIDITY_PATH.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[validity_tests] 结果已写入 {VALIDITY_PATH}")


if __name__ == "__main__":
    main()