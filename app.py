"""
llm-validator — FastAPI Web 应用 (v3.0)
多LLM标注验证系统 · 思路B
用户认证 · 自定义LLM接入 · 上传/分析/下载
"""

import json
import os
import sys
import time
import uuid
import re
import threading
import asyncio

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from validator import AnnotationValidator, ConsensusEngine, AgreementMetrics
from validator.auth import (
    register as auth_register,
    login as auth_login,
    verify_token,
    get_user,
    get_provider_list,
    set_llm_config,
    get_llm_configs,
    get_active_llm_configs,
    delete_llm_config,
    PROVIDER_META,
)
from validator.llm_providers import LLMOrchestrator, FreeSimulatedProvider

# ── App ──
app = FastAPI(title="LLM Annotation Validator", description="多LLM交叉验证标注系统", version="3.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer(auto_error=False)

# ── 路径 ──
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DATA_PATH = DATA_DIR / "sample_data.json"
EXISTING_DATA = Path("/home/osboxes/Desktop/data-annotation/data/student-01_labeled_refined.json")
for d in [DATA_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 会话 ──
sessions: dict = {}
sessions_lock = threading.Lock()
MAX_UPLOAD_MB = 200
MAX_RECORDS = 50000
EXPIRE_HOURS = 6

# ── Auth 依赖 ──
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户（可选认证——未登录也可用免费模式）"""
    if credentials is None:
        return None
    user = verify_token(credentials.credentials)
    return user


async def require_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """要求用户必须登录"""
    user = verify_token(credentials.credentials)
    if not user:
        raise HTTPException(401, "请先登录")
    return user


# ── 工具函数 ──

def _new_session(user_id: Optional[int] = None) -> str:
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {
        "created_at": time.time(), "user_id": user_id,
        "original_filename": "", "records": 0,
        "has_labels": False, "has_llm_results": False,
        "sample_path": "", "result_path": "",
        "samples": [], "gold_labels": {},
        "llm_annotations": {}, "summary": None,
    }
    return sid


def _validate_json_structure(data: list) -> dict:
    if not isinstance(data, list):
        raise HTTPException(400, "JSON 必须是一个数组 (list of records)")
    n = len(data)
    if n == 0:
        raise HTTPException(400, "JSON 数组为空")
    if n > MAX_RECORDS:
        raise HTTPException(400, f"记录数 {n} 超过上限 {MAX_RECORDS}")
    first = data[0]
    has_title = bool(first.get("question_title") or first.get("title"))
    has_content = bool(first.get("question_content") or first.get("content"))
    has_labels = bool(
        first.get("labels", {}).get("label") if isinstance(first.get("labels"), dict)
        else first.get("label")
    )
    if not has_content and not has_title:
        raise HTTPException(400, "数据格式不符合要求。每条记录需包含 title 和 content 字段")
    return {"records": n, "has_title": has_title, "has_content": has_content, "has_labels": has_labels}


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^\w\.\-]", "_", name)[:128]


def _cleanup_old_sessions():
    now = time.time()
    with sessions_lock:
        expired = [sid for sid, s in sessions.items() if now - s["created_at"] > EXPIRE_HOURS * 3600]
        for sid in expired:
            for p in ["sample_path", "result_path"]:
                fp = sessions[sid].get(p)
                if fp and os.path.exists(fp):
                    try: os.remove(fp)
                    except OSError: pass
            del sessions[sid]


# ══════════════════════════════════════════
# Auth API
# ══════════════════════════════════════════

@app.post("/api/auth/register")
async def register(username: str = Form(...), password: str = Form(...), display_name: str = Form("")):
    if len(username) < 2 or len(username) > 32:
        raise HTTPException(400, "用户名长度 2-32 个字符")
    if len(password) < 6:
        raise HTTPException(400, "密码至少 6 个字符")
    result = auth_register(username, password, display_name)
    if not result["ok"]:
        raise HTTPException(409, result["error"])
    return {"ok": True, "message": "注册成功，请登录"}


@app.post("/api/auth/login")
async def login(username: str = Form(...), password: str = Form(...)):
    result = auth_login(username, password)
    if not result["ok"]:
        raise HTTPException(401, result["error"])
    return result


@app.get("/api/auth/me")
async def me(user: dict = Depends(require_user)):
    info = get_user(user["user_id"])
    if not info:
        raise HTTPException(404, "用户不存在")
    configs = get_llm_configs(user["user_id"])
    return {"user": dict(info), "llm_configs": configs}


# ══════════════════════════════════════════
# LLM 配置 API
# ══════════════════════════════════════════

@app.get("/api/providers")
async def list_providers():
    """获取所有可用的 LLM 提供商列表"""
    return get_provider_list()


@app.get("/api/user/configs")
async def get_configs(user: dict = Depends(require_user)):
    """获取用户的 LLM 配置"""
    configs = get_llm_configs(user["user_id"])
    # 隐藏 API key 中间部分
    for c in configs:
        key = c.get("api_key", "")
        if len(key) > 8:
            c["api_key_masked"] = key[:6] + "*" * (len(key) - 8) + key[-2:]
        elif key:
            c["api_key_masked"] = "****"
        else:
            c["api_key_masked"] = ""
    return {"configs": configs}


@app.post("/api/user/configs")
async def save_config(
    provider: str = Form(...),
    api_key: str = Form(""),
    model: str = Form(""),
    base_url: str = Form(""),
    label: str = Form(""),
    user: dict = Depends(require_user),
):
    """保存 LLM 配置"""
    result = set_llm_config(user["user_id"], provider, api_key, model, base_url, label)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return {"ok": True, "message": f"{provider} 配置已保存"}


@app.post("/api/user/custom-config")
async def add_custom_config(
    label: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(""),
    base_url: str = Form(...),
    user: dict = Depends(require_user),
):
    """添加新的自定义端点"""
    from validator.auth import add_custom_config as _add_custom
    result = _add_custom(user["user_id"], label, api_key, model, base_url)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@app.delete("/api/user/configs/detail/{config_id}")
async def delete_config_by_id(config_id: int, user: dict = Depends(require_user)):
    """按 ID 删除配置"""
    from validator.auth import delete_llm_config_by_id
    result = delete_llm_config_by_id(user["user_id"], config_id)
    return {"ok": True}


@app.delete("/api/user/configs/{provider}")
async def delete_config(provider: str, user: dict = Depends(require_user)):
    """按提供商名删除配置"""
    result = delete_llm_config(user["user_id"], provider)
    return {"ok": True}


@app.post("/api/user/configs/{provider}/test")
async def test_config(
    provider: str,
    api_key: str = Form(""),
    model: str = Form(""),
    base_url: str = Form(""),
    user: dict = Depends(require_user),
):
    """测试 LLM 配置连接"""
    from validator.llm_providers import test_llm_connection
    result = await test_llm_connection(provider, api_key, model, base_url)
    return result


# ══════════════════════════════════════════
# Status & Data
# ══════════════════════════════════════════

@app.get("/api/status")
async def status():
    return {
        "status": "ok", "version": "3.0.0",
        "project": "LLM Annotation Validator (思路B)",
        "description": "多LLM交叉验证标注系统 · 支持用户上传·自定义LLM·免费兜底",
        "auth_available": True,
        "max_upload_mb": MAX_UPLOAD_MB, "max_records": MAX_RECORDS,
        "session_expire_hours": EXPIRE_HOURS,
        "demo_available": os.path.exists(SAMPLE_DATA_PATH),
        "providers": len(PROVIDER_META),
    }


# ══════════════════════════════════════════
# Upload & Analyze
# ══════════════════════════════════════════

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    sample_size: str = Form("200"),
    user: dict = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(400, "仅支持 .json 文件")
    raw = await file.read()
    if len(raw) / (1024*1024) > MAX_UPLOAD_MB:
        raise HTTPException(400, f"文件过大，上限 {MAX_UPLOAD_MB}MB")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"JSON 解析失败: {e}")

    info = _validate_json_structure(data)
    uid = user["user_id"] if user else None
    sid = _new_session(uid)
    safe_name = _safe_filename(file.filename)
    sample_path = str(UPLOAD_DIR / f"{sid}_samples.json")

    # 采样逻辑
    import random
    random.seed(int(time.time()))
    total = len(data)
    if sample_size == "all":
        n = min(total, MAX_RECORDS)
        sampled = data[:n]  # 全量不随机
    else:
        n = min(int(sample_size), total)
        sampled = random.sample(data, n)

    samples = []
    gold_labels = {}
    for i, item in enumerate(sampled):
        title = item.get("question_title", item.get("title", ""))
        content = item.get("question_content", item.get("content", ""))
        raw_label = (
            item.get("labels", {}).get("label", "") if isinstance(item.get("labels"), dict)
            else item.get("label", "")
        )
        samples.append({"idx": i, "title": str(title)[:80], "content": str(content)[:500], "gold_label": str(raw_label)})
        if raw_label:
            gold_labels[str(i)] = str(raw_label)

    with sessions_lock:
        sessions[sid].update({
            "original_filename": safe_name, "records": info["records"],
            "has_labels": info["has_labels"], "sample_path": sample_path,
            "samples": samples, "gold_labels": gold_labels,
        })

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples, "gold_labels": gold_labels, "total_samples": len(samples),
                    "total_records": info["records"], "has_labels": info["has_labels"],
                    "original_filename": safe_name}, f, ensure_ascii=False, indent=2)

    _cleanup_old_sessions()
    return {"session_id": sid, "filename": safe_name, "total_records": info["records"],
            "sampled": len(samples), "has_labels": info["has_labels"],
            "samples": samples[:20], "message": f"上传成功！已采样 {len(samples)} 条。"
            + (" 数据含标注标签，可进行验证分析。" if info["has_labels"] else " 数据不含标注标签，仅展示概览。")}


@app.post("/api/analyze/{session_id}")
async def analyze_session(
    session_id: str,
    provider_ids: str = Query("", description="逗号分隔的配置 ID 列表，为空则使用全部"),
    user: dict = Depends(get_current_user),
):
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")

    # 解析选中的 provider ID 列表
    selected_ids = None
    if provider_ids:
        try:
            selected_ids = [int(x) for x in provider_ids.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "provider_ids 格式错误")

    # 初始化进度
    with sessions_lock:
        sess["progress"] = {"status": "starting", "current": 0, "total": 0, "llm": ""}
        sess["has_llm_results"] = False

    # 启动后台分析
    asyncio.create_task(_run_analysis(session_id, user, selected_ids))

    return {"status": "started", "session_id": session_id, "message": "分析已启动"}


async def _run_analysis(session_id: str, user: dict, selected_provider_ids: list = None):
    """后台运行 LLM 分析"""
    try:
        with sessions_lock:
            sess = sessions.get(session_id)
        if not sess:
            return

        samples = sess["samples"]
        gold_labels = sess["gold_labels"]

        uid = sess.get("user_id") or (user["user_id"] if user else None)
        user_configs = []
        if uid:
            user_configs = get_active_llm_configs(uid, selected_provider_ids)
            if not user_configs:
                user_configs = [{"provider": "free", "label": "Free (Simulated)", "api_key": "", "model": "", "base_url": ""}]

        # 进度回调
        async def on_progress(current, total, llm_name):
            with sessions_lock:
                s = sessions.get(session_id)
                if s:
                    s["progress"] = {
                        "status": "running",
                        "current": current,
                        "total": total,
                        "llm": llm_name,
                    }

        with sessions_lock:
            s = sessions.get(session_id)
            if s:
                s["progress"] = {"status": "running", "current": 0, "total": 0, "llm": ""}

        orchestrator = LLMOrchestrator(user_configs)
        llm_results = await orchestrator.run_parallel_validation(samples, on_progress)

        with sessions_lock:
            s = sessions.get(session_id)
            if s:
                s["has_llm_results"] = True
                s["llm_annotations"] = llm_results
                s["progress"]["status"] = "computing"
                # 保存 used_providers 供 results 端点使用
                s["used_providers"] = [
                    {"provider": cfg["provider"], "name": cfg.get("label", cfg["provider"])}
                    for cfg in (user_configs or [{"provider": "free", "label": "Free (Simulated)"}])
                ]

        llm_names = list(llm_results.keys())

        validator = AnnotationValidator()
        validator.results_cache["llm_annotations"] = llm_results
        validator.results_cache["gold_labels"] = gold_labels
        validator.results_cache["samples_count"] = len(samples)
        validator.results_cache["samples"] = samples
        summary = validator.compute_summary()

        with sessions_lock:
            s = sessions.get(session_id)
            if s:
                s["summary"] = summary

        result_path = str(UPLOAD_DIR / f"{session_id}_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with sessions_lock:
            s = sessions.get(session_id)
            if s:
                s["result_path"] = result_path
                s["progress"] = {"status": "done", "current": 1, "total": 1, "llm": ""}
    except Exception as e:
        with sessions_lock:
            s = sessions.get(session_id)
            if s:
                s["progress"] = {"status": "error", "current": 0, "total": 0, "llm": str(e)[:100]}


@app.get("/api/analyze/{session_id}/progress")
async def get_analyze_progress(session_id: str):
    """获取分析进度"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    progress = dict(sess.get("progress", {}))
    progress["has_llm_results"] = sess.get("has_llm_results", False)
    progress["has_summary"] = sess.get("summary") is not None
    return progress


@app.get("/api/analyze/{session_id}/results")
async def get_analyze_results(session_id: str):
    """获取分析完成后的结果"""
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    if not sess.get("has_llm_results"):
        raise HTTPException(400, "尚未完成分析")
    if not sess.get("summary"):
        raise HTTPException(400, "分析尚未完成")

    summary = sess["summary"]
    samples = sess["samples"]
    gold_labels = sess["gold_labels"]
    llm_results = sess["llm_annotations"]
    llm_names = list(llm_results.keys())

    result = dashboard_data(summary, samples, gold_labels, llm_results, llm_names)
    result["used_providers"] = sess.get("used_providers", [])
    return result


@app.get("/api/results/{session_id}")
async def get_results(session_id: str):
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    if not sess.get("has_llm_results"):
        raise HTTPException(400, "尚未进行分析")
    result = dashboard_data(sess["summary"], sess["samples"], sess["gold_labels"],
                          sess["llm_annotations"], list(sess["llm_annotations"].keys()))
    result["used_providers"] = sess.get("used_providers", [])
    return result


@app.get("/api/download/{session_id}")
async def download_results(session_id: str):
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
    with sessions_lock:
        sess = sessions.get(session_id)
    if not sess:
        raise HTTPException(404, "会话不存在或已过期")
    return {"session_id": session_id, "filename": sess["original_filename"],
            "records": sess["records"], "has_labels": sess["has_labels"],
            "has_llm_results": sess["has_llm_results"],
            "created_at": sess.get("created_at", 0),
            "expires_in_hours": round(EXPIRE_HOURS - (time.time() - sess["created_at"]) / 3600, 1)}


@app.get("/api/demo")
async def load_demo(user: dict = Depends(get_current_user)):
    if not os.path.exists(SAMPLE_DATA_PATH):
        if os.path.exists(EXISTING_DATA):
            AnnotationValidator.generate_demo_data(str(EXISTING_DATA), str(SAMPLE_DATA_PATH))
        else:
            raise HTTPException(404, "无可用数据")

    with open(SAMPLE_DATA_PATH) as f:
        demo = json.load(f)

    sid = _new_session(user["user_id"] if user else None)
    with sessions_lock:
        sessions[sid].update({
            "has_labels": True, "has_llm_results": True,
            "llm_annotations": demo["llm_annotations"],
            "gold_labels": demo["gold_labels"],
            "samples_count": demo["total_samples"],
            "samples": demo["samples"],
            "original_filename": "student-01 (演示数据)",
        })

    validator = AnnotationValidator()
    validator.results_cache.update({
        "llm_annotations": demo["llm_annotations"],
        "gold_labels": demo["gold_labels"],
        "samples_count": demo["total_samples"],
        "samples": demo["samples"],
    })
    summary = validator.compute_summary()
    with sessions_lock:
        sessions[sid]["summary"] = summary

    result_path = str(UPLOAD_DIR / f"{sid}_result.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with sessions_lock:
        sessions[sid]["result_path"] = result_path

    return {"session_id": sid, "dashboard": dashboard_data(
        summary, demo["samples"], demo["gold_labels"], demo["llm_annotations"], demo["llm_names"])}


# ══════════════════════════════════════════
# Static & Frontend
# ══════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Dashboard not found</h1>")


static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ══════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════

def dashboard_data(summary, samples, gold_labels, llm_annotations, llm_list):
    has_gold = bool(gold_labels)
    return {
        "overview": {
            "total_samples": summary["total_samples"],
            "llm_count": summary["llm_count"],
            "llm_list": llm_list,
            "fleiss_kappa": summary.get("fleiss_kappa", 0),
            "overall_agreement": summary.get("percentage_agreement", {}).get("overall", 0),
            "majority_agreement": summary.get("percentage_agreement", {}).get("majority", 0),
            "consensus_vs_gold": summary.get("consensus_vs_gold", {}).get("accuracy", 0) if has_gold else None,
            "has_gold_labels": has_gold,
        },
        "llm_performance": [
            {"name": llm, "accuracy": round(stats["accuracy"] * 100, 1),
             "correct": stats["correct"], "total": stats["total"]}
            for llm, stats in summary.get("llm_vs_gold", {}).items()
        ],
        "pairwise_matrix": summary.get("pairwise_agreement", {}),
        "llm_ranking": [{"llm": r["llm"], "accuracy": r["accuracy"] * 100}
                        for r in summary.get("llm_ranking", [])],
        "consensus_detail": summary.get("consensus_vs_gold", {}),
        "label_distribution": summary.get("label_breakdown", {}).get("label_distribution", {}),
        "level_distribution": summary.get("label_breakdown", {}).get("level_distribution", {}),
        "samples": samples[:50],
        "consensus_samples": [{"idx": k, "consensus_label": v["majority_label"],
                               "gold_label": gold_labels.get(k, "?"),
                               "agreement_ratio": v["agreement_ratio"],
                               "is_consensus": v["is_consensus"], "votes": v["llm_votes"]}
                              for k, v in summary.get("consensus_results", {}).items()][:50],
        "agreement_tiers": {
            "high": sum(1 for s in (summary.get("consensus_results", {}) or {}).values()
                        if s.get("agreement_ratio", 0) >= 0.8),
            "medium": sum(1 for s in (summary.get("consensus_results", {}) or {}).values()
                          if 0.5 <= s.get("agreement_ratio", 0) < 0.8),
            "low": sum(1 for s in (summary.get("consensus_results", {}) or {}).values()
                       if 0 < s.get("agreement_ratio", 0) < 0.5),
            "none": sum(1 for s in (summary.get("consensus_results", {}) or {}).values()
                        if s.get("agreement_ratio", 0) == 0),
        },
    }


if __name__ == "__main__":
    import uvicorn

    if os.path.exists(EXISTING_DATA) and not os.path.exists(SAMPLE_DATA_PATH):
        print("正在生成演示数据...")
        AnnotationValidator.generate_demo_data(str(EXISTING_DATA), str(SAMPLE_DATA_PATH))
        print("演示数据生成完成！")

    print("=" * 50)
    print("  LLM 标注验证系统 v3.0")
    print("  用户认证 · 自定义 LLM 接入 · 免费兜底")
    print("=" * 50)
    print(f"  访问: http://localhost:8900")
    print(f"  上传: {MAX_UPLOAD_MB}MB / {MAX_RECORDS} 条")
    print(f"  会话: {EXPIRE_HOURS} 小时过期")
    print("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8900)
