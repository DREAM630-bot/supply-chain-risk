# 四维批判规则文档（Feedback Rules）

本文件是 **Feedback Agent 的 System Prompt 核心依据**，定义四维批判规则的操作化定义。

---

## 零、流水线位置与职责边界

```
Extractor ──→ Feedback ──→ Revision ──→ 聚合 / Shapley 归因
```

| 智能体 | 输入 | 输出 | 禁止行为 |
|:------|:-----|:-----|:---------|
| Extractor | 舆情事件字典（opinion_corpus.md） | evidence_anchor、severity_raw、scope_word 等 | — |
| **Feedback** | Extractor 输出 + 本规则文档 | triggered_rules、action、score_delta | **不直接输出最终分** |
| Revision | severity_raw + score_delta | severity_final（clamp 修正后） | 不重写 severity_raw 的依据 |

**硬约束**：Feedback 只输出 `triggered_rules`、`action`（下调/上调/维持/驳回）和 `score_delta`，不直接输出最终分；Revision 依据 `score_delta` 做 clamp 修正。

---

## 一、输出契约（Output Contract）

Feedback 对每条事件的输出必须严格符合以下 JSON 结构：

```json
{
  "event_id": "E001-2025-09-08-南方都市报",
  "triggered_rules": ["R3", "R4"],
  "action": "下调",
  "score_delta": -0.25,
  "rule_details": {
    "R1": {"triggered": false, "reason": "evidence_anchor 逐字命中原文"},
    "R3": {"triggered": true, "reason": "距窗口终点 12.7 个月 > 6 个月", "delta": -0.15},
    "R4": {"triggered": true, "reason": "单一数据点'激增2.3倍'且 severity_raw=0.70 > 0.60", "delta": -0.10}
  }
}
```

- `action` 取值：`驳回` / `下调` / `维持` / `上调`
- `score_delta`：数值（驳回时为 `null`）
- `rule_details` 中每条规则须给出 `triggered` 与 `reason`，触发时附 `delta`

---

## 二、四维规则操作化定义

### R1 证据一致性（硬门禁）

| 项 | 定义 |
|:---|:-----|
| 触发条件 | Extractor 输出的 `evidence_anchor` **不能**在舆情事件字典对应事件的「原文」字段中逐字存在 |
| 动作 | `action = "驳回"`，`score_delta = null`，该事件不进入后续评分与归因 |
| 操作化判定 | 去除首尾空白后的**严格字符串包含匹配**；不允许同义改写、缩写、省略号截断或 paraphrase |
| 门禁性质 | **硬门禁**：一旦触发，R2–R4 不再评估，直接驳回 |

> 设计意图：防止 LLM 在抽取阶段产生幻觉引用，确保每一个严重度评分都有可回溯的逐字证据锚点。

---

### R2 范围词校准

| 项 | 定义 |
|:---|:-----|
| 触发条件 | `severity_raw` 超出其 `scope_word` 对应的上限（见下表） |
| 动作 | `action = "下调"`，`score_delta = -(severity_raw - 上限)`，即拉回上限 |

**范围词-严重度上限表**：

| scope_word | severity_raw 上限 | 说明 |
|:----------:|:-----------------:|:-----|
| 局部 | **0.70** | 单点/单环节脉冲事件，不应超过 0.70 |
| 区域 | **0.85** | 跨环节区域性事件，上限 0.85 |
| 全局 | **1.00** | 全链路系统性事件，方可接近 1.0 |

> 示例：scope_word="局部"、severity_raw=0.80 → 触发，score_delta = -(0.80 - 0.70) = **-0.10**。

---

### R3 时效衰减

| 项 | 定义 |
|:---|:-----|
| 触发条件 | 事件日期距**窗口终点（2026-09-30）超过 6 个月**，即事件日期早于 **2026-03-30** |
| 动作 | `action = "下调"`，`score_delta ∈ [-0.20, -0.10]`（LLM 在区间内依据事件残余影响程度选取） |
| 区间选取指引 | 距窗口终点 6–9 个月 → 偏轻（-0.10 附近）；9 个月以上 → 偏重（-0.20 附近） |

**与 `time_decay_flag` 的交互（操作注记）**：
- `time_decay_flag = false` 的事件为**结构性/持续性风险**（如饲料涨价、高温干旱），其影响不随时间衰减，**建议对本规则豁免**；
- `time_decay_flag = true` 的事件为脉冲式舆情，正常适用本规则。

---

### R4 单因子一致性

| 项 | 定义 |
|:---|:-----|
| 触发条件 | `severity_raw` 的量化依据**仅来自单一数据点**，且 `severity_raw > 0.60` |
| 动作 | `action = "下调"`，`score_delta ∈ [-0.15, -0.05]`（LLM 在区间内依据数据点代表性选取） |
| 「单一数据点」操作化 | 原文中可提取的量化指标**仅 1 个**（如仅有"激增2.3倍"或仅有"下降15%"），无第二个独立量化指标交叉印证 |
| 区间选取指引 | 数据点代表性弱（如社交平台发帖）→ 偏重（-0.15）；代表性强（如官方抽检数据）→ 偏轻（-0.05） |

> 设计意图：防止单一孤证驱动高分，要求高分必须有多源/多指标交叉支撑。

---

## 三、规则组合与 action 判定逻辑

```
1. R1 触发？
   ├─ 是 → action = 驳回，score_delta = null，终止（R2–R4 不评估）
   └─ 否 ↓
2. R2/R3/R4 是否至少一条触发？
   ├─ 否 → action = 维持，score_delta = 0
   └─ 是 → score_delta = Σ 各触发规则的 delta
           （建议累计下限 -0.30，防止过度压制）
           ├─ score_delta < 0 → action = 下调
           ├─ score_delta > 0 → action = 上调（当前规则集不产生正向 delta，保留扩展位）
           └─ score_delta = 0 → action = 维持
```

**规则独立性**：R2、R3、R4 相互独立，可同时触发，delta 累加。

---

## 四、Revision 的 clamp 修正机制

Revision 接收 Feedback 的 `score_delta` 后，按以下公式修正：

```
severity_final = clamp(severity_raw + score_delta, 0.0, 1.0)

clamp(x, lo, hi):
    x < lo → lo
    x > hi → hi
    否则   → x
```

- `action = "驳回"` 的事件：Revision 不产出 `severity_final`，该事件从样本中剔除；
- 其余事件：Revision 仅做 clamp，**不重写** severity_raw 的依据与 reasoning。

---

## 五、工作示例（基于 opinion_corpus.md 实际事件）

**例 1：E001-2025-09-08-南方都市报**（N5，scope=局部，decay=true，2025-09-08）
- R1：evidence_anchor 逐字命中 → 不触发
- R2：假设 severity_raw=0.70，未超局部上限 0.70 → 不触发
- R3：距窗口终点 2026-09-30 约 12.7 个月 > 6 个月，且 decay=true → **触发**，delta=-0.15
- R4：单一数据点"激增2.3倍"且 severity_raw=0.70 > 0.60 → **触发**，delta=-0.10
- 结果：`triggered_rules=["R3","R4"]`，`action="下调"`，`score_delta=-0.25`
- Revision：severity_final = clamp(0.70 - 0.25, 0, 1) = **0.45**

**例 2：E004-2026-07-22-新华社**（N1，scope=区域，decay=false，2026-07-22）
- R1：evidence_anchor 逐字命中 → 不触发
- R2：假设 severity_raw=0.80，未超区域上限 0.85 → 不触发
- R3：距窗口终点约 2.3 个月 < 6 个月，且 decay=false → 不触发
- R4：原文含"下降约15%""饲料短缺""高价外购"多因子 → 不触发
- 结果：`triggered_rules=[]`，`action="维持"`，`score_delta=0`
- Revision：severity_final = clamp(0.80 + 0, 0, 1) = **0.80**

**例 3：R1 驳回示例**（假设 Extractor 幻觉）
- Extractor 输出 evidence_anchor="牧场产奶量暴跌30%"，但原文为"产奶量环比下降约15%"
- R1：逐字匹配失败 → **触发**，`action="驳回"`，`score_delta=null`，事件剔除

---

## 六、机器可读版本（Python dict）

以下 Python dict 供 Feedback Agent 的 System Prompt 构建或后端校验模块直接使用。

```python
FEEDBACK_RULES = {
    "version": "1.0",
    "pipeline": ["Extractor", "Feedback", "Revision"],
    "output_contract": {
        "fields": ["event_id", "triggered_rules", "action", "score_delta", "rule_details"],
        "action_enum": ["驳回", "下调", "维持", "上调"],
        "constraint": "Feedback 不直接输出最终分；Revision 依据 score_delta 做 clamp 修正",
    },
    "rules": {
        "R1": {
            "name": "证据一致性",
            "type": "hard_gate",
            "trigger": "evidence_anchor 未在舆情事件字典原文中逐字存在",
            "match_method": "strip() + 严格字符串包含",
            "action": "驳回",
            "score_delta": None,
            "blocks": ["R2", "R3", "R4"],
        },
        "R2": {
            "name": "范围词校准",
            "type": "soft",
            "trigger": "severity_raw 超出 scope_word 上限",
            "caps": {"局部": 0.70, "区域": 0.85, "全局": 1.00},
            "action": "下调",
            "score_delta_formula": "-(severity_raw - cap)",
        },
        "R3": {
            "name": "时效衰减",
            "type": "soft",
            "trigger": "事件日期距窗口终点超过 6 个月",
            "window_end": "2026-09-30",
            "threshold_months": 6,
            "cutoff_date": "2026-03-30",
            "action": "下调",
            "score_delta_range": [-0.20, -0.10],
            "exemption": "time_decay_flag=false 的事件建议豁免",
        },
        "R4": {
            "name": "单因子一致性",
            "type": "soft",
            "trigger": "量化依据仅单一数据点 且 severity_raw > 0.60",
            "action": "下调",
            "score_delta_range": [-0.15, -0.05],
        },
    },
    "combination": {
        "delta_aggregation": "sum",
        "delta_floor": -0.30,
        "revision_clamp": "severity_final = clamp(severity_raw + score_delta, 0.0, 1.0)",
        "reject_behavior": "驳回事件不产出 severity_final，从样本中剔除",
    },
}
```