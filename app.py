"""
供应链风险全景测度系统 — 暴露度 × 责任度四象限仪表盘 + 风险智能问答
"""

import json
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(page_title="供应链风险全景测度系统", layout="wide")

# ==========================================
# 2. 账号密码登录拦截
# ==========================================
VALID_USERNAME = "admin123456"       # 账号
VALID_PASSWORD = "qwe123456"         # 密码

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("供应链风险全景测度系统")
    st.info("请输入账号和密码：")
    
    col1, col2 = st.columns(2)
    with col1:
        user_input = st.text_input("账号")
    with col2:
        pass_input = st.text_input("密码", type="password")
        
    if st.button("登录系统", type="primary"):
        if user_input.strip() == VALID_USERNAME and pass_input.strip() == VALID_PASSWORD:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("账号或密码错误，请重试。")
            
    st.stop()

# ==========================================
# 3. 登录成功后的主系统代码
# ==========================================
st.sidebar.success(f"欢迎登录！当前用户：{VALID_USERNAME}")

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = PROJECT_ROOT / "data" / "results.json"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from shapley_engine import build_graph, compute_max_flow, TOPOLOGY  # noqa: E402

# ==========================================
# 大模型配置
# ==========================================
try:
    API_KEY = st.secrets["API_KEY"] 
except Exception:
    API_KEY = "你的API_KEY"

BASE_URL = "https://api.deepseek.com" 
MODEL_NAME = "deepseek-chat" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY and "你的" not in API_KEY else None

# ---- 四象限配置 ----------------------------------------------------------
QUADRANT_STYLE = {
    "显性核心": {"color": "#E74C3C", "strategy": "高暴露 × 高责任：应优先配置应急资源、建立专项预案。"},
    "隐性关键": {"color": "#F39C12", "strategy": "低暴露 × 高责任：属『沉默瓶颈』，需结构加固与冗余备份。"},
    "局部扰动": {"color": "#3498DB", "strategy": "高暴露 × 低责任：以监控预警为主，防止过度响应。"},
    "常规节点": {"color": "#27AE60", "strategy": "低暴露 × 低责任：纳入常规观察与周期性复核即可。"},
}

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
    if not RESULTS_PATH.exists():
        st.error(f"未找到 {RESULTS_PATH}")
        st.stop()
    return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

def build_quadrant_chart(nodes: list, median_exposure: float, median_responsibility: float) -> go.Figure:
    df = pd.DataFrame(nodes)
    fig = px.scatter(
        df, x="exposure", y="responsibility", color="quadrant",
        color_discrete_map={k: v["color"] for k, v in QUADRANT_STYLE.items()},
        hover_name="node_name", hover_data={"node": True, "exposure": ":.2f", "responsibility": ":.4f", "quadrant": True},
        title="暴露度 × 责任度四象限分布", labels={"exposure": "暴露度", "responsibility": "责任度", "quadrant": "象限"},
    )
    fig.add_vline(x=median_exposure, line_dash="dash", line_color="gray", annotation_text=f"暴露度中位数 {median_exposure:.3f}")
    fig.add_hline(y=median_responsibility, line_dash="dash", line_color="gray", annotation_text=f"责任度中位数 {median_responsibility:.4f}")
    fig.update_layout(xaxis=dict(range=[0, 1.05]), yaxis=dict(range=[0, max(df["responsibility"].max() * 1.4, 0.25)]), height=520)
    for _, row in df.iterrows():
        fig.add_annotation(x=row["exposure"], y=row["responsibility"], text=f"{row['node']}", showarrow=True, arrowcolor="#888")
    return fig

# ==========================================
# 调用大模型（流式输出）
# ==========================================
def ask_llm_stream(prompt: str, context_data: str):
    if not client:
        yield "系统提示：检测到未配置大模型 API Key，当前无法进行深度智能推理。"
        return
        
    system_prompt = f"""
    你是一个顶级的供应链风险管理专家。你的系统使用了“最大流博弈”和“Shapley值”来量化风险。
    当前网络是单链拓扑（N1牧场 -> N2工厂 -> N3分仓 -> N4冷链 -> N5配送 -> N6终端）。
    在单链拓扑中，任何一个节点停摆都会导致全网瘫痪，因此各节点的 Shapley 责任度理论值均等（1/6≈0.1667）。
    但是，本系统引入了“拓扑连通性惩罚因子”，使得上游核心转换节点（如N2）和源头节点（如N1）的责任度略高于末端节点。
    
    这里为你提供底层数据的上下文：
    {context_data}
    
    回答要求：
    1. 你必须结合用户的提问，给出明确的业务决策或解释。不要当复读机。
    2. 如果用户问为什么某个节点责任度低却被判为高责任，请从“低暴露度（沉默炸弹）”的角度解释。
    3. 如果用户问末端节点（如N5/N6）断供，请解释“末端停摆不会逆流切断上游，且网络有冗余，属局部扰动”。
    4. 如果用户对比多个节点，请直接给出“该把钱给谁”的建议。
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"系统提示：调用大模型失败（{str(e)}），请检查网络或 API Key。"

# ---- 智能问答路由 ------------------------------
def render_chat_tab(data: dict):
    st.subheader("供应链风险智能问答")
    
    if not client:
        st.warning("当前未配置大模型 API Key，系统将使用离线默认逻辑。请在代码中填入 API_KEY。")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("您可以尝试问以下问题：\n\n"
                    "1. 台风导致 N5 停摆，N2 也设备老化，预算给谁？\n\n"
                    "2. N6 大面积关店是致命风险吗？\n\n"
                    "3. N2 暴露度只有 0.50，为什么它的责任度依然在 0.17 以上？它属于隐性关键节点吗？")

    if st.session_state.messages:
        if st.button("清空对话历史"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("请输入您的供应链决策问题...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            p = prompt.lower()
            nodes = data.get("nodes", [])
            node_map = {n["node"]: n for n in nodes}
            
            found_nodes = []
            for alias, nid in NODE_ALIASES.items():
                if alias in prompt and nid not in found_nodes:
                    found_nodes.append(nid)

            if any(k in p for k in ["排名", "谁最", "排序"]):
                ranked = sorted(nodes, key=lambda n: n["responsibility"], reverse=True)
                lines = ["节点责任度排名（降序）：\n"]
                for i, n in enumerate(ranked, 1):
                    lines.append(f"{i}. {n['node']} {n['node_name']} — 责任度 {n['responsibility']:.4f}｜象限：{n['quadrant']}")
                reply = "\n".join(lines)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop() 
                
            elif any(k in p for k in ["全面战争", "全部瘫痪", "全部毁灭"]):
                reply = "针对宏观系统性风险的回应：\n\n如果全网瘫痪，最大流归零，Shapley 值失效。本系统定位于局部灾害的级联失效测算，不适用于全面战争。"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop()
                
            elif any(k in p for k in ["shapley", "最大流", "归因", "怎么算"]):
                reply = "Shapley 拓扑归因说明：\n\n我们使用最大流作为特征函数，通过 5000 次蒙特卡罗采样计算边际贡献。在单链拓扑下，所有节点分摊断流责任。"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.stop()

            if len(found_nodes) == 1:
                n = node_map.get(found_nodes[0], {})
                context_data = f"当前查询节点：{n.get('node')} {n.get('node_name')}，角色：{n.get('role')}，暴露度：{n.get('exposure')}，责任度：{n.get('responsibility')}，象限：{n.get('quadrant')}。"
                
                anchors = "、".join(f"「{a}」" for a in n.get("evidence_anchor", [])) or "无"
                rules = "、".join(n.get("feedback_rules", [])) or "未触发任何规则"
                st.markdown(
                    f"**节点详情：{n['node']} {n['node_name']}**\n\n"
                    f"- 暴露度：{n['exposure']:.2f} | 责任度：{n['responsibility']:.4f} | 象限：{n['quadrant']}\n"
                    f"- 证据锚点：{anchors}\n"
                    f"- 触发规则：{rules}\n\n"
                    f"**大模型业务解读：**"
                )
                
                with st.spinner("智能体正在推理..."):
                    full_reply = st.write_stream(ask_llm_stream(prompt, context_data))
                    
                history_content = f"**节点详情：{n['node']} {n['node_name']}**\n\n- 暴露度：{n['exposure']:.2f} | 责任度：{n['responsibility']:.4f} | 象限：{n['quadrant']}\n- 证据锚点：{anchors}\n- 触发规则：{rules}\n\n**大模型业务解读：**\n{full_reply}"
                st.session_state.messages.append({"role": "assistant", "content": history_content})

            elif len(found_nodes) >= 2:
                n1 = node_map.get(found_nodes[0], {})
                n2 = node_map.get(found_nodes[1], {})
                context_data = f"当前对比两个节点。节点A：{n1.get('node')} (暴露度{n1.get('exposure')}, 责任度{n1.get('responsibility')}, {n1.get('quadrant')})；节点B：{n2.get('node')} (暴露度{n2.get('exposure')}, 责任度{n2.get('responsibility')}, {n2.get('quadrant')})。"
                st.markdown("**智能体决策建议：**")
                
                with st.spinner("智能体正在推理..."):
                    full_reply = st.write_stream(ask_llm_stream(prompt, context_data))
                    
                history_content = f"**智能体决策建议：**\n{full_reply}"
                st.session_state.messages.append({"role": "assistant", "content": history_content})

            else:
                reply = "抱歉，我无法理解。您可以尝试问：“台风导致 N5 停摆，N2 也设备老化，预算给谁？”"
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})


def main():
    st.title("供应链风险全景测度系统")

    data = load_results()
    nodes = data.get("nodes", [])
    if not nodes:
        st.error("results.json 中无 nodes 数据。")
        st.stop()

    median_exposure = data.get("median_exposure", float(pd.Series([n["exposure"] for n in nodes]).median()))
    median_responsibility = data.get("median_responsibility", float(pd.Series([n["responsibility"] for n in nodes]).median()))

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

        st.subheader(f"二、节点详情：{sel['node']} {sel['node_name']}")
        with st.expander("Evidence Anchor 原文", expanded=True):
            anchors = sel.get("evidence_anchor", [])
            if anchors:
                for a in anchors: st.markdown(f"- `{a}`")
            else: st.info("无证据锚点")
        with st.expander("管理策略"):
            st.markdown(QUADRANT_STYLE.get(sel['quadrant'], {}).get('strategy', ''))

    with tab2:
        render_chat_tab(data)

if __name__ == "__main__":
    main()
