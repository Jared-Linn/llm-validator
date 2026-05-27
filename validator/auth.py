"""
User Authentication — 用户认证系统
SQLite + bcrypt + JWT
"""

import os
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt

# ── 配置 ──
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "users.db")
JWT_SECRET = os.environ.get("JWT_SECRET", uuid.uuid4().hex + uuid.uuid4().hex)
JWT_ALGO = "HS256"
JWT_EXPIRE_HOURS = 72

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """初始化数据库表"""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL,
            display_name TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS llm_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            provider TEXT NOT NULL,    -- 'openai' | 'anthropic' | 'deepseek' | 'gemini' | 'custom'
            model TEXT NOT NULL DEFAULT '',
            api_key TEXT NOT NULL DEFAULT '',
            base_url TEXT NOT NULL DEFAULT '',
            is_active INTEGER DEFAULT 1,
            UNIQUE(user_id, provider)
        );
    """)
    conn.commit()
    conn.close()


# ── 用户管理 ──

def register(username: str, password: str, display_name: str = "") -> dict:
    """注册新用户"""
    conn = _get_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE username=?", (username,)
        ).fetchone()
        if existing:
            return {"ok": False, "error": "用户名已存在"}

        pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        now = time.time()
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, created_at, display_name) VALUES (?,?,?,?)",
            (username, pw_hash, now, display_name or username),
        )
        user_id = cur.lastrowid
        conn.commit()
        return {"ok": True, "user_id": user_id}
    finally:
        conn.close()


def login(username: str, password: str) -> dict:
    """登录，返回 JWT"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, password_hash, display_name FROM users WHERE username=?",
            (username,),
        ).fetchone()
        if not row:
            return {"ok": False, "error": "用户名或密码错误"}

        if not bcrypt.checkpw(password.encode(), row["password_hash"].encode()):
            return {"ok": False, "error": "用户名或密码错误"}

        token = jwt.encode(
            {
                "user_id": row["id"],
                "username": row["username"],
                "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRE_HOURS),
                "iat": datetime.now(timezone.utc),
            },
            JWT_SECRET,
            algorithm=JWT_ALGO,
        )
        return {
            "ok": True,
            "token": token,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "display_name": row["display_name"],
            },
        }
    finally:
        conn.close()


def verify_token(token: str) -> Optional[dict]:
    """验证 JWT，返回 user info 或 None"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return {"user_id": payload["user_id"], "username": payload["username"]}
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_user(user_id: int) -> Optional[dict]:
    """获取用户信息"""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, display_name, created_at FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)
    finally:
        conn.close()


# ── LLM 配置管理 ──

PROVIDER_META = {
    "openai": {
        "label": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o-mini",
        "default_base_url": "https://api.openai.com/v1",
        "needs_key": True,
        "docs_url": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "models": ["claude-sonnet-4", "claude-haiku-3-5", "claude-opus-4"],
        "default_model": "claude-haiku-3-5",
        "default_base_url": "https://api.anthropic.com/v1",
        "needs_key": True,
        "docs_url": "https://console.anthropic.com/settings/keys",
    },
    "deepseek": {
        "label": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
        "default_base_url": "https://api.deepseek.com/v1",
        "needs_key": True,
        "docs_url": "https://platform.deepseek.com/api_keys",
    },
    "gemini": {
        "label": "Google Gemini",
        "models": ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro"],
        "default_model": "gemini-2.0-flash",
        "default_base_url": "https://generativelanguage.googleapis.com/v1beta",
        "needs_key": True,
        "docs_url": "https://aistudio.google.com/app/apikey",
    },
    "custom": {
        "label": "自定义 (OpenAI 兼容)",
        "models": [],
        "default_model": "",
        "default_base_url": "",
        "needs_key": True,
        "docs_url": None,
    },
    "free": {
        "label": "免费内置模型",
        "models": ["free-simulated"],
        "default_model": "free-simulated",
        "default_base_url": "",
        "needs_key": False,
        "docs_url": None,
    },
}


def get_provider_list() -> list:
    """获取可用的提供商列表"""
    return [
        {"id": k, **v}
        for k, v in PROVIDER_META.items()
    ]


def set_llm_config(user_id: int, provider: str, api_key: str = "",
                   model: str = "", base_url: str = "") -> dict:
    """保存或更新 LLM 配置"""
    meta = PROVIDER_META.get(provider)
    if not meta:
        return {"ok": False, "error": f"不支持的提供商: {provider}"}

    if meta["needs_key"] and not api_key:
        return {"ok": False, "error": f"{meta['label']} 需要填写 API Key"}

    model = model or meta.get("default_model", "")
    base_url = base_url or meta.get("default_base_url", "")

    conn = _get_conn()
    try:
        conn.execute("""
            INSERT INTO llm_configs (user_id, provider, model, api_key, base_url)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider) DO UPDATE SET
                model=excluded.model,
                api_key=excluded.api_key,
                base_url=excluded.base_url,
                is_active=1
        """, (user_id, provider, model, api_key, base_url))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


def get_llm_configs(user_id: int) -> list:
    """获取用户的所有 LLM 配置"""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT provider, model, api_key, base_url, is_active FROM llm_configs WHERE user_id=?",
            (user_id,),
        ).fetchall()
        configs = []
        for row in rows:
            cfg = dict(row)
            meta = PROVIDER_META.get(cfg["provider"], {})
            cfg["label"] = meta.get("label", cfg["provider"])
            configs.append(cfg)
        return configs
    finally:
        conn.close()


def get_active_llm_configs(user_id: int) -> list:
    """获取用户已激活的 LLM 配置（有 API key 的）"""
    all_configs = get_llm_configs(user_id)
    active = [c for c in all_configs if c["api_key"] and c["is_active"]]
    # 如果没有配置任何 LLM，返回空列表（前端会 fallback 到 free）
    return active


def delete_llm_config(user_id: int, provider: str) -> dict:
    """删除某个 LLM 配置"""
    conn = _get_conn()
    try:
        conn.execute(
            "DELETE FROM llm_configs WHERE user_id=? AND provider=?",
            (user_id, provider),
        )
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


# 初始化
init_db()
