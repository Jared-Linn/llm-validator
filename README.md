# 🧠 LLM 标注验证系统 (思路B · v3.0)

## 多LLM交叉验证 · 共识投票 · Web 看板

用多个大语言模型独立验证同一批标注数据，通过共识投票机制判断标注质量。

- **GitHub:** https://github.com/Jared-Linn/llm-validator
- **在线演示:** http://jared-linn.asia:8900

---

## 功能总览

| 功能 | 说明 |
|------|------|
| ✅ 用户认证 | 注册/登录，每个用户独立管理 LLM 配置 |
| ✅ 6 种提供商 | OpenAI / Anthropic / DeepSeek / Gemini / 自定义 / 免费内置 |
| ✅ 多自定义端点 | 支持添加任意数量的 OpenAI 兼容 API（vLLM / Ollama / 中转API） |
| ✅ 模型选择 | 分析前可选哪些模型参与验证 |
| ✅ 采样选项 | 上传时可选 200/500/1000 条或全量分析 |
| ✅ 进度条 | 实时显示分析进度（当前 LLM、批次、百分比） |
| ✅ 无标签分析 | 无 gold label 数据也能跑 LLM，展示共识与一致率 |
| ✅ 测试按钮 | 配置页一键测试 API 连通性 |
| ✅ 模型自定义输入 | 模型名支持下拉预设 + 自由输入 |
| ✅ 看板可视化 | Chart.js 仪表盘：Fleiss Kappa、热力图、排名、饼图 |
| ✅ Docker 部署 | docker-compose 一键启动，健康检查 |
| ✅ 下载结果 | 分析结果导出为 JSON |

---

## 核心流程

```
用户上传 JSON 数据 (含/不含 gold label)
       │
       ▼
  选择采样量 (200/500/1000/全量)
       │
       ▼
  选择参与模型 (多选弹窗)
       │
       ▼
  ┌──────────────────────────────┐
  │  OpenAI (GPT-4o)             │
  │  Anthropic (Claude Sonnet 4) │  ← 并行调用多LLM
  │  DeepSeek (V4 Flash)         │
  │  自定义端点 1 / 2 / ...      │
  └──────┬───────────────────────┘
         │
         ▼
  进度条实时更新 (每批次完成)
         │
         ▼
  共识引擎
   - 多数投票 / Fleiss' Kappa
   - Pairwise Agreement 矩阵
   - 各LLM vs Gold (如有标签)
         │
         ▼
  📊 Web 看板 — 实时可视化
```

---

## 项目结构

```
llm-validator/
├── app.py                     # FastAPI Web 服务 (端口 8900)
├── Dockerfile                 # Docker 构建
├── docker-compose.yml         # 容器编排 (含健康检查)
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量模板
├── healthcheck.py             # Docker 健康检查
├── validator/
│   ├── auth.py                # 用户认证 + LLM 配置管理 (SQLite/JWT/bcrypt)
│   ├── engine.py              # 验证引擎 (采样 → 派发 → 汇总)
│   ├── consensus.py           # 共识引擎 (多数投票/加权/Fleiss' Kappa)
│   ├── metrics.py             # 评估指标 (Cohen/Fleiss/百分比一致)
│   ├── llm_providers.py       # 多LLM API 接入 + 连接测试 + 进度回调
│   └── llm_clients.py         # LLM 客户端注册表 (预留)
├── static/
│   └── index.html             # Web 前端 (Chart.js + 无框架)
├── data/
│   ├── sample_data.json       # 演示数据 (自动生成)
│   └── uploads/               # 用户上传文件存储
└── README.md
```

---

## 快速启动

### 本地运行

```bash
pip install -r requirements.txt
python3 app.py
# 打开 http://localhost:8900
```

### Docker 部署

```bash
# 首次构建
docker compose build

# 启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止
docker compose down
```

---

## 配置 LLM 提供商

登录后进入 **⚙️ 模型配置** 页面：

### 内置模型
| 提供商 | 模型 | API Key 获取 |
|--------|------|-------------|
| OpenAI | gpt-4o, gpt-4o-mini, ... | [platform.openai.com](https://platform.openai.com/api-keys) |
| Anthropic | claude-sonnet-4, claude-haiku-3-5, ... | [console.anthropic.com](https://console.anthropic.com/settings/keys) |
| DeepSeek | deepseek-v4-flash, deepseek-v4-pro | [platform.deepseek.com](https://platform.deepseek.com/api_keys) |
| Gemini | gemini-2.0-flash, gemini-2.0-pro, ... | [aistudio.google.com](https://aistudio.google.com/app/apikey) |

### 自定义端点
支持任意 OpenAI 兼容 API（中转服务 / vLLM / Ollama / 本地模型），可添加多个：
1. 填写**名称**（如 `中转API`）
2. 填写 **API Key** 和**接口地址**
3. 填写**模型名**
4. 点击 **🧪 测试** 验证连通性
5. 保存后即可在分析时选用

### 免费内置
无需配置，基于已有标注的模拟验证（准确率 ~82%）。当没有配置其他 LLM 时自动启用。

---

## 使用流程

### 1. 上传数据
支持 JSON 格式，每条记录需包含 `title` 和 `content` 字段。
如果有 gold label，格式为 `labels: { "label": "1.7" }`。

### 2. 选择采样量
上传区可选：200条 / 500条 / 1000条 / 全量

### 3. 选择模型
点"开始验证 →"后，如果有多个已配置模型，弹出多选框。
只勾选你想参与分析的模型。

### 4. 查看进度
实时进度条显示：
- 当前正在调用的 LLM
- 已完成批次 / 总批次
- 百分比

### 5. 分析结果
| 看板模块 | 说明 |
|---------|------|
| 📊 概览卡片 | 样本数、LLM数、Fleiss Kappa、一致率 |
| 🏆 LLM排名 | 各LLM vs 原始标签的准确率 (有gold时) |
| 🔄 两两一致矩阵 | LLM间的 pairwise agreement 热力图 |
| 🎯 一致性分层 | 高/中/低一致性样本分布饼图 |
| 📋 样本明细 | 逐条展示原始标签、共识标签、投票详情 |

---

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册 |
| `/api/auth/login` | POST | 登录 (返回 JWT) |
| `/api/providers` | GET | 获取可用提供商列表 |
| `/api/user/configs` | GET/POST | 管理 LLM 配置 |
| `/api/user/custom-config` | POST | 添加自定义端点 |
| `/api/user/configs/{provider}/test` | POST | 测试 API 连通性 |
| `/api/upload` | POST | 上传数据 (支持 sample_size 参数) |
| `/api/analyze/{id}` | POST | 启动分析 (支持 provider_ids 参数) |
| `/api/analyze/{id}/progress` | GET | 轮询进度 |
| `/api/analyze/{id}/results` | GET | 获取结果 |
| `/api/download/{id}` | GET | 下载结果 JSON |

---

## 关键指标

| 指标 | 说明 |
|------|------|
| **Fleiss' Kappa** | 多标注者一致性（>0.6 良好，>0.8 优秀） |
| **Pairwise Agreement** | 每对LLM的标注一致率 |
| **Consensus vs Gold** | 多LLM共识与原始标注的一致率（需 gold label） |
| **Majority Agreement** | ≥半数的LLM达成一致的比例 |
| **Overall Agreement** | 所有LLM完全一致的比例 |

---

## 后续计划

- [ ] 混淆矩阵和错误分析
- [ ] 多轮对话标注支持
- [ ] 在线修正：看板中直接修改标签并重新计算
- [ ] 导出 PDF/HTML 验证报告
- [ ] LLM 历史表现追踪与加权投票

---

*滇池学院理工学院 · NLP 方向 · 基于 Hermes Agent 构建*
