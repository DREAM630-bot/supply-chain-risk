"""
供应链风险全景测度系统 — 暴露度 × 责任度四象限仪表盘 + 风险智能问答
==================================================================
启动方式：
    cd supply-chain-risk-attribution
    streamlit run app.py

Tab 1：四象限仪表盘（暴露度 × 责任度散点图 + 节点详情）
Tab 2：风险智能问答（基于 data/results.json 与 shapley_engine 的规则问答）

数据源：data/results.json（由 scripts/shapley_engine.py 与 Controller 流水线产出）
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results.json"

# 确保可导入 scripts 目录下的模块（Shapley 代码节点）
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from shapley_engine import build_graph, compute_max_flow, TOPOLOGY  # noqa: E402

# ---- 四象限配置 ----------------------------------------------------------
QUADRANT_STYLE = {
    "显性核心": {
        "color": "#E74C3C",
        "strategy": (
            "高暴露 × 高责任：该节点既是舆情高风险区又是流量瓶颈，"
            "应优先配置应急资源、建立专项预案，实施每日风险例会与冗余备份。"
        ),
    },
    "隐性关键": {
        "color": "#F39C12",
        "strategy": (
            "低暴露 × 高责任：当前舆情平静但拓扑责任重大，属『沉默瓶颈』，"
            "需结构加固与冗余备份，定期开展压力测试，防止风险突发传导。"
        ),
    },
    "局部扰动": {
        "color": "#3498DB",
        "strategy": (
            "高暴露 × 低责任：舆情活跃但拓扑影响有限，以监控预警为主，"
            "防止风险沿链路传导升级，配置舆情自动告警与快速响应机制。"
        ),
    },
    "常规节点": {
        "color": "#27AE60",
        "strategy": (
            "低暴露 × 低责任：风险与责任均较低，纳入常规观察与周期性复核即可，"
            "按季度评估即可，无需专项投入。"
        ),
    },
}

# ---- 节点别名（问答中的节点识别）------------------------------------------
NODE_ALIASES = {
    "N1": "N1", "敕勒川": "N1", "牧场": "N1", "奶源": "N1",
    "N2": "N2", "呼和浩特": "N2", "工厂": "N2",
    "N3": "N3", "武汉": "N3", "分仓": "N3", "云仓": "N3", "仓储": "N3",
    "N4": "N4", "冷链": "N4", "干线": "N4",
    "N5": "N5", "华南": "N5", "配送": "N5", "末端": "N5",
    "N6": "N6", "零售": "N6", "终端": "N6", "销售": "N6",
}


@st.cache_data
def load_results() -> dict:
    """读取 data/results.json。"""
    if not RESULTS_PATH.exists():
        st.error(f"未找到 {RESULTS_PATH}，请先运行 python scripts/shapley_engine.py")
        st.stop()
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))


def build_quadrant_chart(nodes: list, median_exposure: float,
                         median_responsibility: float) -> go.Figure:
    """构建四象限散点图。"""
    df = pd.DataFrame(nodes)
    fig = px.scatter(
        df, x="exposure", y="responsibility", color="quadrant",
        color_discrete_map={k: v["color"] for k, v in QUADRANT_STYLE.items()},
        hover_name="node_name",
        hover_data={"node": True, "exposure": ":.2f",
                    "responsibility": ":.4f", "quadrant": True},
        title="暴露度 × 责任度四象限分布",
        labels={"exposure": "暴露度 (Exposure)",
                "responsibility": "责任度 (Responsibility)",
                "quadrant": "象限"},
    )
    fig.add_vline(x=median_exposure, line_dash="dash", line_color="gray",
                  annotation_text=f"暴露度中位数 {median_exposure:.3f}",
                  annotation_position="top left")
    fig.add_hline(y=median_responsibility, line_dash="dash", line_color="gray",
                  annotation_text=f"责任度中位数 {median_responsibility:.4f}",
                  annotation_position="top right")
    fig.update_layout(
        xaxis=dict(range=[0, 1.05], gridcolor="#EEE"),
        yaxis=dict(range=[0, max(df["responsibility"].max() * 1.4, 0.25)],
                   gridcolor="#EEE"),
        legend_title_text="象限", height=520,
    )
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["exposure"], y=row["responsibility"],
            text=f"{row['node']} {row['node_name']}",
            showarrow=True, arrowhead=1, arrowsize=0.8,
            arrowcolor="#888", font=dict(size=10), yshift=12,
        )
    return fig


def detect_node(prompt: str):
    """从问题中识别节点 ID（N1–N6），未识别返回 None。"""
    for alias, nid in NODE_ALIASES.items():
        if alias in prompt:
            return nid
    return None


# ---- 智能问答 ------------------------------------------------------------

def generate_chat_response(prompt: str, data: dict) -> str:
    """根据用户问题路由到对应数据源，生成自然语言回复（规则路由，非 LLM）。"""
    p = prompt.lower()
    nodes = data.get("nodes", [])
    node_map = {n["node"]: n for n in nodes}

    # ① 排名 / 谁最危险
    if any(k in p for k in ["排名", "最危险", "谁最", "排序", "责任度最高",
                            "风险最高", "最严重", "top"]):
        ranked = sorted(nodes, key=lambda n: n["responsibility"], reverse=True)
        lines = ["📊 **节点责任度排名（降序）**：\n"]
        for i, n in enumerate(ranked, 1):
            lines.append(
                f"{i}. **{n['node']} {n['node_name']}** — "
                f"责任度 {n['responsibility']:.4f}｜暴露度 {n['exposure']:.2f}"
                f"｜象限：{n['quadrant']}"
            )
        top = ranked[0]
        lines.append(
            f"\n⚠️ 责任度最高的是 **{top['node']} {top['node_name']}**"
            f"（{top['responsibility']:.4f}），属于「{top['quadrant']}」。"
        )
        return "\n".join(lines)

    # ② Shapley / 最大流 / 计算原理（调用代码节点）
    if any(k in p for k in ["shapley", "最大流", "归因", "责任度来源",
                            "怎么算", "如何计算", "原理", "计算方法"]):
        G = build_graph()
        flow_value, _ = compute_max_flow(G, TOPOLOGY["source"], TOPOLOGY["sink"])
        resp = data.get("responsibility", {})
        n_samples = data.get("n_samples", "N/A")
        resp_str = "、".join(f"{k}={v:.4f}" for k, v in resp.items())
        return (
            "🔬 **Shapley 拓扑归因说明**：\n\n"
            f"- 特征函数 v(S)：联盟 S 诱导子图上 {TOPOLOGY['source']}→"
            f"{TOPOLOGY['sink']} 的**最大流**（实时计算值 = {flow_value:.2f}）。\n"
            f"- 蒙特卡罗估计：N={n_samples} 次随机排列，逐节点计算边际贡献 "
            f"v(S∪{{i}}) − v(S) 的均值后归一化到 Σ=1。\n"
            f"- 当前各节点责任度：{resp_str}\n\n"
            "📐 串行链拓扑性质：理论上各节点 Shapley 值相等（= 最大流/节点数），"
            "结果中的微小差异来自蒙特卡罗估计噪声。"
        )

    # ③ 节点详情 / 为什么 / 证据
    nid = detect_node(prompt)
    if nid and nid in node_map:
        n = node_map[nid]
        anchors = "、".join(f"「{a}」" for a in n.get("evidence_anchor", [])) or "无"
        rules = "、".join(n.get("feedback_rules", [])) or "未触发任何规则"
        ev_lines = [
            f"  - {e['event_id']}：原始分 {e['severity_raw']} → "
            f"修正分 {e['severity_final']}"
            for e in n.get("events", [])
        ]
        ev_str = "\n".join(ev_lines) if ev_lines else "  无事件明细"
        qcfg = QUADRANT_STYLE.get(n["quadrant"], {})
        strategy = qcfg.get("strategy", "")
        return (
            f"📋 **节点详情：{n['node']} {n['node_name']}**（{n.get('role', '')}）\n\n"
            f"- 暴露度 exposure = **{n['exposure']:.2f}**\n"
            f"- 责任度 responsibility = **{n['responsibility']:.4f}**\n"
            f"- 四象限分类：**{n['quadrant']}**\n"
            f"- 证据锚点：{anchors}\n"
            f"- Feedback 触发规则：{rules}\n"
            f"- 事件明细（原始分 → 修正分）：\n{ev_str}\n\n"
            f"🎯 管理策略：{strategy}"
        )

    # ④ 问"为什么/证据"但未指明节点
    if any(k in p for k in ["为什么", "证据", "依据", "原因", "详情"]):
        return (
            "💡 请指明您想查询的节点，例如：\n\n"
            "- 「N1 敕勒川牧场为什么风险高？」\n"
            "- 「N5 华南配送的证据是什么？」\n\n"
            "我会提供该节点的暴露度、责任度、证据锚点、触发规则与事件明细。"
        )

    # ⑤ 兜底
    return (
        "🤔 抱歉，我暂时无法理解这个问题。您可以尝试问：\n\n"
        "1. 「N1 敕勒川牧场的风险详情和证据是什么？」\n"
        "2. 「哪个节点最危险？请给出责任度排名。」\n"
        "3. 「Shapley 责任度和最大流是怎么计算的？」"
    )


def render_chat_tab(data: dict):
    """渲染风险智能问答 Tab。"""
    st.subheader("供应链风险智能问答")

    # 初始化对话历史（同一 session 内持久）
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 对话开始前展示 3 个示例问题
    if not st.session_state.messages:
        st.markdown(
            "**💡 您可以尝试问以下问题：**\n\n"
            "1. N1 敕勒川牧场的风险详情和证据是什么？\n\n"
            "2. 哪个节点最危险？请给出责任度排名。\n\n"
            "3. Shapley 责任度和最大流是怎么计算的？"
        )

    # 清空对话按钮（已有历史时显示）
    if st.session_state.messages:
        if st.button("🗑️ 清空对话历史"):
            st.session_state.messages = []
            st.rerun()

    # 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 接收用户输入
    prompt = st.chat_input("请输入您想查询的供应链风险问题...")
    if prompt:
        # 追加并显示用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # 生成并显示助手回复
        reply = generate_chat_response(prompt, data)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def main():
    st.set_page_config(page_title="供应链风险四象限仪表盘", layout="wide")
    st.title("供应链风险全景测度系统 — 暴露度 × 责任度四象限仪表盘")

    data = load_results()
    st.warning(data.get("disclaimer", ""))
    st.caption(f"网络：{data.get('network_name', '')}　|　"
               f"聚合策略：{data.get('aggregation_strategy', 'max')}　|　"
               f"蒙特卡罗样本数：{data.get('n_samples', 'N/A')}")

    nodes = data.get("nodes", [])
    if not nodes:
        st.error("results.json 中无 nodes 数据，请先运行完整流水线。")
        st.stop()

    median_exposure = data.get("median_exposure")
    median_responsibility = data.get("median_responsibility")
    if median_exposure is None:
        exposures = [n["exposure"] for n in nodes]
        median_exposure = float(pd.Series(exposures).median())
    if median_responsibility is None:
        resps = [n["responsibility"] for n in nodes]
        median_responsibility = float(pd.Series(resps).median())

    # ---- 左侧：节点选择（全局侧边栏，供 Tab 1 使用）----
    with st.sidebar:
        st.header("节点选择")
        node_options = [f"{n['node']} {n['node_name']}" for n in nodes]
        selected_label = st.selectbox("选择节点 N1–N6", node_options)
        selected_idx = node_options.index(selected_label)
        sel = nodes[selected_idx]
        st.metric("暴露度", f"{sel['exposure']:.2f}")
        st.metric("责任度", f"{sel['responsibility']:.4f}")
        st.metric("象限", sel["quadrant"])

    # ---- 双标签页 ----
    tab1, tab2 = st.tabs(["四象限仪表盘", "风险智能问答"])

    # ===== Tab 1：四象限仪表盘 =====
    with tab1:
        st.subheader("一、四象限全景")
        fig = build_quadrant_chart(nodes, median_exposure, median_responsibility)
        st.plotly_chart(fig, use_container_width=True)

        q_counts = {}
        for n in nodes:
            q_counts[n["quadrant"]] = q_counts.get(n["quadrant"], 0) + 1
        cols = st.columns(4)
        for col, (qname, qcfg) in zip(cols, QUADRANT_STYLE.items()):
            col.markdown(
                f"<span style='color:{qcfg['color']};font-weight:bold'>"
                f"● {qname}</span>：{q_counts.get(qname, 0)} 个节点",
                unsafe_allow_html=True,
            )

        st.subheader(f"二、节点详情：{sel['node']} {sel['node_name']}")
        with st.expander("📋 Evidence Anchor 原文", expanded=True):
            anchors = sel.get("evidence_anchor", [])
            if anchors:
                for i, a in enumerate(anchors, 1):
                    st.markdown(f"{i}. `{a}`")
            else:
                st.info("该节点无有效证据锚点。")

        with st.expander("🔍 Feedback 触发的规则列表"):
            rules = sel.get("feedback_rules", [])
            if rules:
                for r in rules:
                    st.markdown(f"- {r}")
            else:
                st.info("该节点未触发任何批判规则（全部维持）。")

        with st.expander("📊 原始分 → 修正分"):
            events = sel.get("events", [])
            if events:
                ev_df = pd.DataFrame([
                    {"事件": e["event_id"], "来源": e.get("source", ""),
                     "原始分 severity_raw": e.get("severity_raw"),
                     "修正分 severity_final": e.get("severity_final"),
                     "修正说明": e.get("revision_note", "")}
                    for e in events
                ])
                st.dataframe(ev_df, use_container_width=True, hide_index=True)
            else:
                st.info("该节点无事件明细。")

        with st.expander("🎯 象限判定与管理策略"):
            qname = sel.get("quadrant", "")
            qcfg = QUADRANT_STYLE.get(qname, {"strategy": "未知象限"})
            st.markdown(f"**象限**：<span style='color:{qcfg.get('color', '#000')};"
                        f"font-weight:bold'>{qname}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**管理策略**：{qcfg['strategy']}")
            st.markdown(
                f"**判定依据**：exposure={sel['exposure']:.2f} "
                f"({'≥' if sel['exposure'] >= median_exposure else '<'} "
                f"中位数 {median_exposure:.3f})，"
                f"responsibility={sel['responsibility']:.4f} "
                f"({'≥' if sel['responsibility'] >= median_responsibility else '<'} "
                f"中位数 {median_responsibility:.4f})"
            )

    # ===== Tab 2：风险智能问答 =====
    with tab2:
        render_chat_tab(data)


if __name__ == "__main__":
    main()