# 🧠 LLM 标注验证系统 (思路B)

## 多LLM交叉验证 · 共识投票 · Web 看板

基于**思路B**构建：用多个大语言模型独立验证同一批标注数据，通过共识投票机制判断标注质量，突破传统 TF-IDF+LR 单模型的 ~85% 瓶颈。

---

## 核心思路

```
原始标注 (pipeline_a / refine_loop)
       │
       ▼
  分层采样 200 条 (兼顾 S1/S2/S3)
       │
       ▼
  ┌──────────────────────────┐
  │   DeepSeek V4 Flash      │
  │   Claude Sonnet 4        │  ← 多个LLM独立验证
  │   Gemini 2.0 Flash       │
  │   GPT-4o (扩展)          │
  └──────┬───────────────────┘
         │
         ▼
  共识引擎 (Consensus Engine)
   - 多数投票
   - 加权投票 (按LLM历史表现)
   - 置信度加权
         │
         ▼
  ┌──────────────────────────┐
  │  Fleiss' Kappa           │
  │  Pairwise Agreement      │  ← 一致性评估指标
  │  LLM vs Gold Standard    │
  │  逐标签层级准确率        │
  └──────────────────────────┘
         │
         ▼
   📊 Web 看板 — 实时可视化
```

---

## 项目结构

```
llm-validator/
├── app.py                     # FastAPI Web 服务 (端口 8900)
├── requirements.txt           # 依赖
├── validator/
│   ├── __init__.py
│   ├── engine.py              # 验证引擎 (加载数据 → 采样 → 派发 → 汇总)
│   ├── consensus.py           # 共识引擎 (多数投票/加权/Fleiss' Kappa)
│   ├── metrics.py             # 评估指标 (Cohen/Fleiss/百分比一致)
│   └── llm_clients.py         # 多LLM客户端注册表
├── data/
│   └── sample_data.json       # 演示验证数据 (自动生成)
├── static/
│   └── index.html             # Web 看板 (Chart.js 可视化)
├── templates/                 # (预留 Jinja2 模板)
└── README.md
```

---

## 快速启动

```bash
# 1. 安装依赖
pip install fastapi uvicorn --break-system-packages

# 2. 启动服务 (自动生成演示数据)
cd /home/osboxes/Desktop/llm-validator
python3 app.py

# 3. 打开浏览器
# http://localhost:8900
```

---

## 看板功能

| 模块 | 内容 |
|------|------|
| 📊 **概览卡片** | 样本数、参与LLM数、Fleiss' Kappa、全一致率、多数一致率、共识vs原始 |
| 🏆 **LLM排名** | 各LLM vs 原始标签的准确率条形排名 |
| 🔄 **两两一致矩阵** | LLM之间的 pairwise agreement 热力图 |
| 🎯 **一致性分层** | 高/中/低一致性样本的分布饼图 |
| 📋 **样本明细** | 逐条展示原始标签、共识标签、匹配状态、投票详情 |

---

## 使用流程

### 1. 准备验证数据

```python
from validator.engine import AnnotationValidator

# 自动从标注数据生成验证集
AnnotationValidator.generate_demo_data(
    "/path/to/labeled_refined.json",  # 输入标注数据
    "data/sample_data.json"           # 输出验证集
)
```

### 2. 派发到真实LLM

看板内置了 HTTP API 用于导入真实 LLM 的验证结果：

```python
import requests

# 获取待验证样本
samples = requests.get("http://localhost:8900/api/dashboard").json()

# 导入某LLM的验证结果
results = [{"idx": 0, "label": "1.7", "confidence": 0.92}]
requests.post("http://localhost:8900/api/llm/deepseek", json=results)
```

### 3. 自定义采样

```bash
# 修改采样参数 (在 app.py 或直接调用 engine)
curl http://localhost:8900/api/validate?n=500&method=stratified
```

---

## 关键指标说明

| 指标 | 说明 |
|------|------|
| **Fleiss' Kappa** | 多标注者一致性（>0.6 良好，>0.8 优秀） |
| **Pairwise Agreement** | 每对LLM的标注一致率 |
| **Consensus vs Gold** | 多LLM共识与原始标注的一致率 |
| **Majority Agreement** | ≥半数的LLM达成一致的比例 |
| **Overall Agreement** | 所有LLM完全一致的比例 |

---

## 扩展：接入真实LLM

系统支持通过 Hermes `delegate_task` 派发标注任务到实际LLM：

```python
# 注册自定义LLM
from validator.llm_clients import LLMClientRegistry, HermesSubAgentClient

LLMClientRegistry.register(HermesSubAgentClient("GPT-4o", model="openai/gpt-4o"))
LLMClientRegistry.register(HermesSubAgentClient("Claude Sonnet 4", model="anthropic/claude-sonnet-4"))
```

然后调用 `app.py` 的验证接口即可。

---

## 后续计划

- [ ] 接入真实 LLM 完成实际验证循环
- [ ] 添加混淆矩阵和错误分析
- [ ] 支持对比多个标注文件的验证结果
- [ ] 在线修正：看板中直接修改标签并重新计算
- [ ] 导出 PDF/HTML 验证报告

---

*滇池学院理工学院 · NLP 方向 · 基于 Hermes Agent 构建*
