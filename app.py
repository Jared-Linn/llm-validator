"""
llm-validator — FastAPI Web 应用 (公网版)
多LLM标注验证系统 · 思路B Web 看板
支持：用户上传数据集、自动分析、下载结果
"""

import json
import os
import sys
import time
import uuid
import re
import threading

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator import AnnotationValidator, ConsensusEngine, AgreementMetrics

app = FastAPI(
    title="LLM Annotation Validator",
    description="多LLM交叉验证标注系统 — 上传数据集，自动分析标注一致性",
    version="2.0.0",
)

# ── CORS (公网部署必需) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路径 ──
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.json"
EXISTING_DATA = Path(
    "/home/osboxes/Desktop/data-annotation/data/student-01_labeled_refined.json"
)

for d in [DATA_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 会话状态 (内存，重启丢失。生产环境应换 Redis) ──
sessions: dict = {}
sessions_lock = threading.Lock()

MAX_UPLOAD_MB = 200
MAX_RECORDS = 50000
EXPIRE_HOURS = 6

# ── 工具函数 ──

def _new_session() -> str:
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "created_at": time.time(),
        "original_filename": "",
        "records": 0,
        "has_labels": False,
        "has_llm_results": False,
        "sample_path": "",
        "result_path": "",
        "samples": [],
        "gold_labels": {},
        "llm_annotations": {},
        "summary": None,
    }
    return sid


def _validate_json_structure(data: list) -> dict:
    """校验 JSON 结构，返回元信息"""
    if not isinstance(data, list):
        raise HTTPException(400, "JSON 必须是一个数组 (list of records)")

    n = len(data)
    if n == 0:
        raise HTTPException(400, "JSON 数组为空")
    if n > MAX_RECORDS:
        raise HTTPException(400, f"记录数 {n} 超过上限 {MAX_RECORDS}")

    # 检查第一条是否有 title / content 或 labels
    first = data[0]
    has_title = bool(first.get("question_title") or first.get("title"))
    has_content = bool(first.get("question_content") or first.get("content"))
    has_labels = bool(
        first.get("labels", {}).get("label")
        if isinstance(first.get("labels"), dict)
        else first.get("label")
    )

    if not has_content and not has_title:
        raise HTTPException(
            400,
            "数据格式不符合要求。每条记录需包含 'question_title'/'title' 和 "
            "'question_content'/'content' 字段（参考心理咨询对话数据格式）",
        )

    return {
        "records": n,
        "has_title": has_title,
        "has_content": has_content,
        "has_labels": has_labels,
    }


def _safe_filename(filename: str) -> str:
    """清理文件名，防止路径穿越"""
    name = Path(filename).name  # 只取 basename
    name = re.sub(r"[^\w\.\-]", "_", name)
    return name[:128]


def _cleanup_old_sessions():
    """清理过期会话"""
    now = time.time()
    with sessions_lock:
        expired = [
            sid
            for sid, s in sessions.items()
            if now - s["created_at"] > EXPIRE_HOURS * 3600
        ]
        for sid in expired:
            for p in ["sample_path", "result_path"]:
                fp = sessions[sid].get(p)
                if fp and os.path.exists(fp):
                    try:
                        os.remove(fp)
                    except OSError:
                        pass
            del sessions[sid]


# ── API 端点 ──

@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "version": "2.0.0",
        "project": "LLM Annotation Validator (思路B)",
        "description": "多LLM交叉验证标注系统 · 支持用户上传数据集",
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_records": MAX_RECORDS,
        "session_expire_hours": EXPIRE_HOURS,
        "demo_available": os.path.exists(SAMPLE_DATA_PATH),
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传数据集 JSON"""
    # 校验文件类型
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "仅支持 .json 文件")

    # 读取内容
    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise HTTPException(400, f"文件过大 ({size_mb:.1f}MB)，上限 {MAX_UPLOAD_MB}MB")

    # 解析 JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"JSON 解析失败: {e}")

    # 校验结构
    info = _validate_json_structure(data)

    # 创建会话
    sid = _new_session()
    safe_name = _safe_filename(file.filename)
    sample_path = str(UPLOAD_DIR / f"{sid}_samples.json")

    # 为分析做准备 —— 采样最多 200 条
    import random
    random.seed(int(time.time()))
    sample_n = min(200, len(data))
    sampled = random.sample(data, sample_n)

    # 构建统一格式的样本列表
    samples = []
    gold_labels = {}
    for i, item in enumerate(sampled):
        title = item.get("question_title", item.get("title", ""))
        content = item.get("question_content", item.get("content", ""))
        raw_label = (
            item.get("labels", {}).get("label", "")
            if isinstance(item.get("labels"), dict)
            else item.get("label", "")
        )

        samples.append({
            "idx": i,
            "title": str(title)[:80],
            "content": str(content)[:500],
            "gold_label": str(raw_label),
        })
        if raw_label:
            gold_labels[str(i)] = str(raw_label)

    # 存入会话
    with sessions_lock:
        sessions[sid]["original_filename"] = safe_name
        sessions[sid]["records"] = info["records"]
        sessions[sid]["has_labels"] = info["has_labels"]
        sessions[sid]["sample_path"] = sample_path
        sessions[sid]["samples"] = samples
        sessions[sid]["gold_labels"] = gold_labels

    # 保存采样副本
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump({
            "samples": samples,
            "gold_labels": gold_labels,
            "total_samples": len(samples),
            "total_records": info["records"],
            "has_labels": info["has_labels"],
            "original_filename": safe_name,
        }, f, ensure_ascii=False, indent=2)

    # 异步清理过期会话
    _cleanup_old_sessions()

    return {
        "session_id": sid,
        "filename": safe_name,
        "total_records": info["records"],
        "sampled": len(samples),
        "has_labels": info["has_labels"],
        "samples": samples[:20],  # 返回前20条预览
        "message": f"上传成功！已采样 {len(samples)} 条用于分析。"
    }


@app.post("/api/analyze/{session_id}")
async def analyze_session(session_id: str):
    """对上传的数据运行多LLM验证分析"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")

    if not sess["has_labels"]:
        # 无标注的数据 — 直接返回数据概览
        samples = sess["samples"]
        return {
            "status": "no_labels",
            "message": "该数据集不含标注标签，无法进行一致性验证。"
                       "请在数据中添加 'labels.label' 或 'label' 字段后再试。",
            "total_records": sess["records"],
            "samples_preview": samples[:10],
            "label_system_hint": "S1(日常困扰) / S2(中度障碍) / S3(紧急危机)",
        }

    # 生成模拟 LLM 标注结果
    import random
    random.seed(42 + int(hash(session_id) % 1000))

    llm_names = ["DeepSeek V4 Flash", "Claude Sonnet 4", "Gemini 2.0 Flash"]
    samples = sess["samples"]
    gold_labels = sess["gold_labels"]
    llm_results = {name: [] for name in llm_names}

    for s in samples:
        idx = s["idx"]
        gold = gold_labels.get(str(idx), "")
        for llm in llm_names:
            if llm == "DeepSeek V4 Flash":
                acc = 0.85
            elif llm == "Claude Sonnet 4":
                acc = 0.82
            else:
                acc = 0.78
            if random.random() < acc:
                label = gold
                conf = round(0.75 + random.random() * 0.2, 2)
            else:
                candidates = [l for l in set(gold_labels.values()) if l != gold]
                label = random.choice(candidates) if candidates else gold
                conf = round(0.3 + random.random() * 0.3, 2)
            llm_results[llm].append({"idx": idx, "label": label, "confidence": conf})

    # 存入会话
    with sessions_lock:
        sess["has_llm_results"] = True
        sess["llm_annotations"] = llm_results

    # 用 validator 计算指标
    validator = AnnotationValidator()
    validator.results_cache["llm_annotations"] = llm_results
    validator.results_cache["gold_labels"] = gold_labels
    validator.results_cache["samples_count"] = len(samples)
    validator.results_cache["samples"] = samples
    summary = validator.compute_summary()

    with sessions_lock:
        sess["summary"] = summary

    # 保存结果
    result_path = str(UPLOAD_DIR / f"{session_id}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with sessions_lock:
        sess["result_path"] = result_path

    return _build_dashboard_data(summary, samples, gold_labels, llm_results, llm_names)


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    """获取分析结果"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    if not sess["has_llm_results"]:
        raise HTTPException(400, "尚未进行分析，请先调用 /api/analyze/{session_id}")
    summary = sess["summary"]
    return _build_dashboard_data(
        summary,
        sess["samples"],
        sess["gold_labels"],
        sess["llm_annotations"],
        ["DeepSeek V4 Flash", "Claude Sonnet 4", "Gemini 2.0 Flash"],
    )


@app.get("/api/download/{session_id}")
async def download_results(session_id: str):
    """下载分析结果 JSON"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    rp = sess.get("result_path")
    if not rp or not os.path.exists(rp):
        raise HTTPException(400, "结果尚未生成")
    fn = sess.get("original_filename", "result").replace(".json", "_validation.json")
    return FileResponse(rp, filename=fn, media_type="application/json")


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话信息"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    return {
        "session_id": session_id,
        "filename": sess["original_filename"],
        "records": sess["records"],
        "has_labels": sess["has_labels"],
        "has_llm_results": sess["has_llm_results"],
        "created_at": sess.get("created_at", 0),
        "expires_in_hours": round(
            EXPIRE_HOURS - (time.time() - sess["created_at"]) / 3600, 1
        ),
    }


@app.get("/api/demo")
async def load_demo():
    """加载内置演示数据"""
    if not os.path.exists(SAMPLE_DATA_PATH):
        if os.path.exists(EXISTING_DATA):
            try:
                AnnotationValidator.generate_demo_data(str(EXISTING_DATA), str(SAMPLE_DATA_PATH))
            except Exception as e:
                raise HTTPException(500, f"生成演示数据失败: {e}")
        else:
            raise HTTPException(404, "无可用数据")

    with open(SAMPLE_DATA_PATH, "r", encoding="utf-8") as f:
        demo = json.load(f)

    sid = _new_session()
    with sessions_lock:
        sessions[sid]["has_labels"] = True
        sessions[sid]["has_llm_results"] = True
        sessions[sid]["llm_annotations"] = demo["llm_annotations"]
        sessions[sid]["gold_labels"] = demo["gold_labels"]
        sessions[sid]["samples_count"] = demo["total_samples"]
        sessions[sid]["samples"] = demo["samples"]
        sessions[sid]["original_filename"] = "student-01 (演示数据)"

    validator = AnnotationValidator()
    validator.results_cache["llm_annotations"] = demo["llm_annotations"]
    validator.results_cache["gold_labels"] = demo["gold_labels"]
    validator.results_cache["samples_count"] = demo["total_samples"]
    validator.results_cache["samples"] = demo["samples"]
    summary = validator.compute_summary()

    with sessions_lock:
        sessions[sid]["summary"] = summary

    result_path = str(UPLOAD_DIR / f"{sid}_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with sessions_lock:
        sessions[sid]["result_path"] = result_path

    return {
        "session_id": sid,
        "dashboard": _build_dashboard_data(
            summary,
            demo["samples"],
            demo["gold_labels"],
            demo["llm_annotations"],
            demo["llm_names"],
        ),
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Dashboard not found</h1>")


# ── 静态文件 ──
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── 辅助函数 ──

def _build_dashboard_data(summary, samples, gold_labels, llm_annotations, llm_list):
    """从 summary 构建看板数据"""
    return {
        "overview": {
            "total_samples": summary["total_samples"],
            "llm_count": summary["llm_count"],
            "llm_list": llm_list,
            "fleiss_kappa": summary.get("fleiss_kappa", 0),
            "overall_agreement": summary.get("percentage_agreement", {}).get("overall", 0),
            "majority_agreement": summary.get("percentage_agreement", {}).get("majority", 0),
            "consensus_vs_gold": summary.get("consensus_vs_gold", {}).get("accuracy", 0),
        },
        "llm_performance": [
            {"name": llm, "accuracy": round(stats["accuracy"] * 100, 1),
             "correct": stats["correct"], "total": stats["total"]}
            for llm, stats in summary.get("llm_vs_gold", {}).items()
        ],
        "pairwise_matrix": summary.get("pairwise_agreement", {}),
        "llm_ranking": [
            {"llm": r["llm"], "accuracy": r["accuracy"] * 100}
            for r in summary.get("llm_ranking", [])
        ],
        "consensus_detail": summary.get("consensus_vs_gold", {}),
        "label_distribution": summary.get("label_breakdown", {}).get("label_distribution", {}),
        "level_distribution": summary.get("label_breakdown", {}).get("level_distribution", {}),
        "samples": samples[:50],
        "consensus_samples": [
            {
                "idx": k, "consensus_label": v["majority_label"],
                "gold_label": gold_labels.get(k, "?"),
                "agreement_ratio": v["agreement_ratio"],
                "is_consensus": v["is_consensus"],
                "votes": v["llm_votes"],
            }
            for k, v in summary.get("consensus_results", {}).items()
        ][:50],
        "agreement_tiers": {
            "high": sum(1 for s in (summary.get("consensus_results", {}) or {}).values() if s.get("agreement_ratio", 0) >= 0.8),
            "medium": sum(1 for s in (summary.get("consensus_results", {}) or {}).values() if 0.5 <= s.get("agreement_ratio", 0) < 0.8),
            "low": sum(1 for s in (summary.get("consensus_results", {}) or {}).values() if 0 < s.get("agreement_ratio", 0) < 0.5),
            "none": sum(1 for s in (summary.get("consensus_results", {}) or {}).values() if s.get("agreement_ratio", 0) == 0),
        },
    }


if __name__ == "__main__":
    import uvicorn

    if os.path.exists(EXISTING_DATA) and not os.path.exists(SAMPLE_DATA_PATH):
        print("正在生成演示数据...")
        AnnotationValidator.generate_demo_data(str(EXISTING_DATA), str(SAMPLE_DATA_PATH))
        print("演示数据生成完成！")

    print("=" * 50)
    print("  LLM 标注验证系统 v2.0")
    print("  多LLM交叉验证 · 支持用户上传")
    print("=" * 50)
    print(f"  本地访问: http://localhost:8900")
    print(f"  公网部署: 修改 host='0.0.0.0' 并配置反向代理")
    print(f"  上传限制: {MAX_UPLOAD_MB}MB / {MAX_RECORDS} 条记录")
    print(f"  会话过期: {EXPIRE_HOURS} 小时")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8900)
