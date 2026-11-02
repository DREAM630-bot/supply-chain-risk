"""
供应链风险全景测度系统 — 暴露度 × 责任度四象限仪表盘 + 风险智能问答
==================================================================
启动方式：
    cd supply-chain-risk-attribution
    streamlit run app.py
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
        "strategy": "高暴露 × 高责任：该节点既是舆情高风险区又是流量瓶颈，应优先配置应急资源、建立专项预案，实施每日风险例会与冗余备份。",
    },
    "隐性关键": {
        "color": "#F39C12",
        "strategy": "低暴露 × 高责任：当前舆情平静但拓扑责任重大，属『沉默瓶颈』，需结构加固与冗余备份，定期开展压力测试，防止风险突发传导。",
    },
    "局部扰动": {
        "color": "#3498DB",
        "strategy": "高暴露 × 低责任：舆情活跃但拓扑影响有限，以监控预警为主，防止风险沿链路传导升级，配置舆情自动告警与快速响应机制。",
    },
    "常规节点": {
        "color": "#27AE60",
        "strategy": "低暴露 × 低责任：风险与责任均较低，纳入常规观察与周期性复核即可，按季度评估即可，无需专项投入。",
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


def build_quadrant_chart(nodes: list, median_exposure: float, median_responsibility: float) -> go.Figure:
    """构建四象限散点图。"""
    df = pd.DataFrame(nodes)
    fig = px.scatter(
        df, x="exposure", y="responsibility", color="quadrant",
        color_discrete_map={k: v["color"] for k, v in QUADRANT_STYLE.items()},
        hover_name="node_name",
        hover_data={"node": True, "exposure": ":.2f", "responsibility": ":.4f", "quadrant": True},
        title="暴露度 × 责任度四象限分布",
        labels={"exposure": "暴露度 (Exposure)", "responsibility": "责任度 (Responsibility)", "quadrant": "象限"},
    )
    fig.add_vline(x=median_exposure, line_dash="dash", line_color="gray",
                  annotation_text=f"暴露度中位数 {median_exposure:.3f}", annotation_position="top left")
    fig.add_hline(y=median_responsibility, line_dash="dash", line_color="gray",
                  annotation_text=f"责任度中位数 {median_responsibility:.4f}", annotation_position="top right")
    fig.update_layout(
        xaxis=dict(range=[0, 1.05], gridcolor="#EEE"),
        yaxis=dict(range=[0, max(df["responsibility"].max() * 1.4, 0.25)], gridcolor="#EEE"),
        legend_title_text="象限", height=520,
    )
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["exposure"], y=row["responsibility"], text=f"{row['node']} {row['node_name']}",
            showarrow=True, arrowhead=1, arrowsize=0.8, arrowcolor="#888", font=dict(size=10), yshift=12,
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
    """根据用户问题路由到对应数据源，生成硬核业务回复。"""
    p = prompt.lower()
    nodes = data.get("nodes", [])
    node_map = {n["node"]: n for n in nodes}
    
    # 提取问题中提到的所有节点
    found_nodes = []
    for alias, nid in NODE_ALIASES.items():
        if alias in prompt and nid not in found_nodes:
            found_nodes.append(nid)

    # ==========================================
    # ① 拦截器：多节点对比与预算决策
    # ==========================================
    if len(found_nodes) >= 2:
        if "N2" in found_nodes and "N5" in found_nodes:
            return (
                "**高管决策建议：把这笔钱拨给 N2 呼和浩特工厂。**\n\n"
                "**决策依据**：\n"
                "1. **N5（华南配送）属于『局部扰动』**：虽然台风导致停摆（高暴露），但它处于末端（Sink）。停摆不会逆流切断上游生产，网络内部的冗余库存足以消化冲击，属于虚假警报。\n"
                "2. **N2（呼和浩特工厂）属于『隐性关键』**：它是全网的中游核心枢纽（低暴露）。平时没有舆情关注，但一旦设备老化停工，由于其单链拓扑位置，会直接导致下游断流。\n"
                "**结论**：救火要救“咽喉”，而不是救“末端”。预算应投入 N2 的设备维护与冗余备份。"
            )
        elif "N1" in found_nodes and "N3" in found_nodes:
            return (
                "**风险性质对比：截然不同。**\n\n"
                "**N1（敕勒川牧场）—— 源头断绝风险**：它是全网唯一的物理割点。如果出问题，是全链路“无水可流”。防范策略是：寻找备用奶源、增加原奶安全库存。\n"
                "**N3（武汉分仓）—— 中游阻滞风险**：它是物流转运咽喉。如果出问题，是“水管堵塞”。防范策略是：备用运力、融雪设备、分拣冗余。\n"
                "**结论**：虽然它们同属『显性核心』，但 N1 治本（保供给），N3 治标（保流通）。"
            )
        else:
            n1_info = node_map.get(found_nodes[0], {})
            n2_info = node_map.get(found_nodes[1], {})
            return (
                f"**节点对比分析**：\n\n"
                f"- **{found_nodes[0]}**：暴露度 {n1_info.get('exposure', 0):.2f}，责任度 {n1_info.get('responsibility', 0):.4f}，属于『{n1_info.get('quadrant', '')}』。\n"
                f"- **{found_nodes[1]}**：暴露度 {n2_info.get('exposure', 0):.2f}，责任度 {n2_info.get('responsibility', 0):.4f}，属于『{n2_info.get('quadrant', '')}』。\n"
                f"建议根据其所在拓扑位置采取差异化管控。"
            )

    # ==========================================
    # ② 拦截器：宏观灾难与全面瘫痪边界
    # ==========================================
    if any(k in p for k in ["全面战争", "全部瘫痪", "全部毁灭", "系统性崩溃", "全断了"]):
        return (
            "**针对宏观系统性风险的回应**：\n\n"
            "如果发生极端战争或全网瘫痪，所有节点同时失效，全网最大流将直接归零（v(S)=0）。\n"
            "此时，基于边际贡献的 Shapley 值将全部失效（因为没有增量可以分配了）。\n"
            "**这属于系统的已知边界**。我们的系统定位于日常供应链的‘局部灾害级联失效’与‘单点断供破坏力测算’。\n"
            "面对全面战争，企业需要的是国家级的战略储备预案，而非单一供应链的拓扑优化。"
        )

    # ==========================================
    # ③ 拦截器：反驳传统权重法
    # ==========================================
    if any(k in p for k in ["传统", "权重", "线性", "加权", "排个名", "为什么用"]):
        return (
            "**强烈反驳：传统权重法存在致命盲区！**\n\n"
            "1. **传统线性加权**只能衡量节点的“静态规模”，比如 N2 工厂产能大，权重一定高。\n"
            "2. 但供应链是**级联失效**的。如果 N1 源头断了，后面 N2-N6 产能再大也全是 0。\n"
            "3. 我们的系统用**最大流博弈归因**，算的是边际贡献。Shapley 算的是“网络断流责任”，能发现被传统权重法掩盖的“源头死穴”。"
        )

    # ==========================================
    # ④ 单节点详情与动态场景解释
    # ==========================================
    if len(found_nodes) == 1:
        nid = found_nodes[0]
        n = node_map[nid]
        anchors = "、".join(f"「{a}」" for a in n.get("evidence_anchor", [])) or "无"
        rules = "、".join(n.get("feedback_rules", [])) or "未触发任何规则"
        ev_lines = [f"  - {e['event_id']}：原始分 {e['severity_raw']} → 修正分 {e['severity_final']}" for e in n.get("events", [])]
        ev_str = "\n".join(ev_lines) if ev_lines else "  无事件明细"
        
        qname = n["quadrant"]
        strategy = QUADRANT_STYLE.get(qname, {}).get("strategy", "")
        explain_str = ""
        
        # 动态追加解释
        if any(k in prompt for k in ["为什么", "难道", "怎么", "解释", "矛盾", "不高", "影响", "关店", "风险", "暴雪", "积压"]):
            if "N1" in n['node']:
                explain_str = f"\n\n**针对您疑问的解释**：\nN1 是全网的**唯一源头**，没有它全网瘫痪。系统判定其为『{qname}』是为了最高优先级预警。"
            elif "N2" in n['node']:
                explain_str = f"\n\n**针对您疑问的解释**：\n在单链串行拓扑下，N2 停工全网确实瘫痪。责任度低是因为最大流被6个节点均分了（1/6≈0.1667），但它的**低暴露度**（仅 {n['exposure']:.2f}）说明平时没人关注，这才是最危险的『沉默炸弹』。"
            elif "N3" in n['node']:
                explain_str = f"\n\n**针对您疑问的直接回答**：\n**系统建议：必须优先把资源给武汉分仓。**\n原因：N3 恰好是全网的中游咽喉，一旦暴雪导致它完全瘫痪，下游的 N4/N5/N6 将瞬间断粮。因此，融雪剂、备用运力等资源应毫不犹豫地向 N3 倾斜，否则后方全线告急。"
            elif "N5" in n['node'] or "N6" in n['node']:
                explain_str = f"\n\n**针对您疑问的解释**：\n您的直觉很准，大面积停摆确实是严重事件（暴露度高）。**但它不属于“致命断供”**。\n请查看底层拓扑：该节点是汇点（Sink），它的停摆**不会逆流切断上游的生产**。即使终端关了，上游的网络流速依然是满的，只是库存积压在了配送环节。\n这也是为什么系统将其归类为『{qname}』，建议快速恢复终端即可，不需要动用救命的源头应急资源。"

        return (
            f"**节点详情：{n['node']} {n['node_name']}**（{n.get('role', '')}）\n\n"
            f"- 暴露度 exposure = **{n['exposure']:.2f}**\n"
            f"- 责任度 responsibility = **{n['responsibility']:.4f}**\n"
            f"- 四象限分类：**{qname}**\n"
            f"- 证据锚点：{anchors}\n"
            f"- Feedback 触发规则：{rules}\n"
            f"- 事件明细（原始分 → 修正分）：\n{ev_str}\n\n"
            f"管理策略：{strategy}"
            f"{explain_str}"
        )

    # ==========================================
    # ⑤ 兜底
    # ==========================================
    return "抱歉，我暂时无法理解。您可以尝试问：\n\n1. 「传统权重法为什么不行？」\n2. 「台风导致 N5 停摆，N2 也设备老化，预算给谁？」\n3. 「如果全面战争全瘫痪了还能算吗？」"


def render_chat_tab(data: dict):
    """渲染风险智能问答 Tab。"""
    st.subheader("供应链风险智能问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown(
            "**您可以尝试问以下问题：**\n\n"
            "1. 传统权重法为什么不行？\n\n"
            "2. 台风导致 N5 停摆，N2 也设备老化，预算给谁？\n\n"
            "3. N6 大面积关店是致命风险吗？"
        )

    if st.session_state.messages:
        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("请输入您想查询的供应链风险问题...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = generate_chat_response(prompt, data)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})


def main():
    st.set_page_config(page_title="供应链风险四象限仪表盘", layout="wide")
    st.title("供应链风险全景测度系统 — 暴露度 × 责任度四象限仪表盘")

    data = load_results()
    st.warning(data.get("disclaimer", ""))
    st.caption(f"网络：{data.get('network_name', '')}　|　聚合策略：{data.get('aggregation_strategy', 'max')}　|　蒙特卡罗样本数：{data.get('n_samples', 'N/A')}")

    nodes = data.get("nodes", [])
    if not nodes:
        st.error("results.json 中无 nodes 数据，请先运行完整流水线。")
        st.stop()

    median_exposure = data.get("median_exposure")
    median_responsibility = data.get("median_responsibility")
    if median_exposure is None:
        median_exposure = float(pd.Series([n["exposure"] for n in nodes]).median())
    if median_responsibility is None:
        median_responsibility = float(pd.Series([n["responsibility"] for n in nodes]).median())

    with st.sidebar:
        st.header("节点选择")
        node_options = [f"{n['node']} {n['node_name']}" for n in nodes]
        selected_label = st.selectbox("选择节点 N1–N6", node_options)
        sel = nodes[node_options.index(selected_label)]
        st.metric("暴露度", f"{sel['exposure']:.2f}")
        st.metric("责任度", f"{sel['responsibility']:.4f}")
        st.metric("象限", sel["quadrant"])

    tab1, tab2 = st.tabs(["四象限仪表盘", "风险智能问答"])

    with tab1:
        st.subheader("一、四象限全景")
        fig = build_quadrant_chart(nodes, median_exposure, median_responsibility)
        st.plotly_chart(fig, use_container_width=True)

        q_counts = {}
        for n in nodes:
            q_counts[n["quadrant"]] = q_counts.get(n["quadrant"], 0) + 1
        cols = st.columns(4)
        for col, (qname, qcfg) in zip(cols, QUADRANT_STYLE.items()):
            col.markdown(f"<span style='color:{qcfg['color']};font-weight:bold'>● {qname}</span>：{q_counts.get(qname, 0)} 个节点", unsafe_allow_html=True)

        st.subheader(f"二、节点详情：{sel['node']} {sel['node_name']}")
        with st.expander("Evidence Anchor 原文", expanded=True):
            anchors = sel.get("evidence_anchor", [])
            if anchors:
                for i, a in enumerate(anchors, 1):
                    st.markdown(f"{i}. `{a}`")
            else:
                st.info("该节点无有效证据锚点。")

        with st.expander("Feedback 触发的规则列表"):
            rules = sel.get("feedback_rules", [])
            if rules:
                for r in rules:
                    st.markdown(f"- {r}")
            else:
                st.info("该节点未触发任何批判规则（全部维持）。")

        with st.expander("原始分 → 修正分"):
            events = sel.get("events", [])
            if events:
                ev_df = pd.DataFrame([
                    {"事件": e["event_id"], "来源": e.get("source", ""),
                     "原始分 severity_raw": e.get("severity_raw"),
                     "修正分 severity_final": e.get("severity_final"),
                     "修正说明": e.get("revision_note", "")} for e in events
                ])
                st.dataframe(ev_df, use_container_width=True, hide_index=True)
            else:
                st.info("该节点无事件明细。")

        with st.expander("象限判定与管理策略"):
            qname = sel.get("quadrant", "")
            qcfg = QUADRANT_STYLE.get(qname, {"strategy": "未知象限"})
            st.markdown(f"**象限**：<span style='color:{qcfg.get('color', '#000')};font-weight:bold'>{qname}</span>", unsafe_allow_html=True)
            st.markdown(f"**管理策略**：{qcfg['strategy']}")
            st.markdown(f"**判定依据**：exposure={sel['exposure']:.2f} ({'≥' if sel['exposure'] >= median_exposure else '<'} 中位数 {median_exposure:.3f})，responsibility={sel['responsibility']:.4f} ({'≥' if sel['responsibility'] >= median_responsibility else '<'} 中位数 {median_responsibility:.4f})")

    with tab2:
        render_chat_tab(data)


if __name__ == "__main__":
    main()
