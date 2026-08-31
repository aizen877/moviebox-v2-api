"""
Moviebox Unofficial v2 API  -  FastAPI gateway for HuggingFace Spaces & Render.

Wraps the MovieBox H5 REST backend (h5-api.aoneroom.com) and exposes clean JSON endpoints.
Features automatic guest Bearer token acquisition, HTTP/2 multiplexing, search suggestions,
catalog filtering, metadata details, high-speed direct stream extraction, and subtitle links.
"""

import asyncio
import contextvars
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from enum import Enum
from urllib.parse import quote, unquote

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse


class FastJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS)

class SubjectType(int, Enum):
    ALL = 0
    MOVIES = 1
    TV_SERIES = 2
    EDUCATION = 3
    MUSIC = 4
    ANIME = 5

HOST_URL = "https://h5-api.aoneroom.com"
REFERER = "https://moviebox.ph"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moviebox_v2_api")

# --- fast HTTP layer & Upstream URLs -----------------------------------------

BASE = HOST_URL.rstrip("/")
API_BASE = f"{BASE}/wefeed-h5api-bff"
REC_BASE = "https://h5.aoneroom.com"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Referer": "https://moviebox.ph/",
    "Origin": "https://moviebox.ph",
    "X-Client-Info": '{"timezone":"Asia/Dhaka"}',
    "X-Request-Lang": "en",
    "Accept": "application/json",
    "Content-Type": "application/json",
    "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "cross-site",
}

PLAYER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "X-Client-Info": '{"timezone":"Asia/Dhaka"}',
    "X-Source": "",
    "sec-ch-ua": '"Chromium";v="148", "Google Chrome";v="148", "Not/A)Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}

# Shared ultra-high-throughput connection pool
_LIMITS = httpx.Limits(max_keepalive_connections=100, max_connections=200, keepalive_expiry=60.0)
_TIMEOUT = httpx.Timeout(15.0, connect=4.0)

# Global bearer token cache & lock
_bearer_token: str | None = None
_token_lock = asyncio.Lock()

# Ultra-fast in-memory TTL cache: key -> (expires_at, value)
_CACHE: dict[str, tuple[float, object]] = {}
HOMEPAGE_TTL = 300.0    # 5 min
SEARCH_TTL = 180.0      # 3 min
DETAILS_TTL = 600.0     # 10 min
METADATA_TTL = 86400.0  # 24 hours (for TMDB ID, Logo, Backdrop)
RECOMMEND_TTL = 600.0   # 10 min


def _cache_get(key: str):
    hit = _CACHE.get(key)
    if hit and hit[0] > time.time():
        return hit[1]
    if hit:
        _CACHE.pop(key, None)
    return None


def _cache_set(key: str, value, ttl: float):
    _CACHE[key] = (time.time() + ttl, value)


async def _get_bearer_token(force_refresh: bool = False) -> str:
    """Auto-acquire a guest JWT token from the x-user response header or cookie."""
    global _bearer_token
    if _bearer_token and not force_refresh:
        return _bearer_token

    async with _token_lock:
        if _bearer_token and not force_refresh:
            return _bearer_token

        try:
            client = getattr(app.state, "client", None)
            if client is None:
                client = httpx.AsyncClient(http2=True, headers=DEFAULT_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
                should_close = True
            else:
                should_close = False

            resp = await client.get(f"{API_BASE}/home?host=moviebox.ph", headers=DEFAULT_HEADERS)
            x_user = resp.headers.get("x-user")
            if x_user:
                try:
                    _bearer_token = json.loads(x_user).get("token")
                except Exception:
                    pass

            if not _bearer_token:
                cookie = resp.headers.get("set-cookie", "")
                m = re.search(r"token=([^;]+)", cookie)
                if m:
                    _bearer_token = m.group(1)

            if should_close:
                await client.aclose()

            logger.info(f"Bearer token initialized/refreshed: {bool(_bearer_token)}")
        except Exception as e:
            logger.error(f"Failed to acquire bearer token: {e}")

    return _bearer_token or ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        http2=True,
        headers=DEFAULT_HEADERS,
        limits=_LIMITS,
        timeout=_TIMEOUT,
        follow_redirects=True
    )
    logger.info("Shared HTTP/2 httpx connection pool initialized.")
    await _get_bearer_token()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="MovieBox API Pro",
    description="Full Pure REST API for moviebox.ph — Ultra High Performance",
    version="2.3.0",
    lifespan=lifespan,
    default_response_class=FastJSONResponse,
)

client_ip_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("client_ip", default=None)


@app.middleware("http")
async def performance_and_ip_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else None
    token = client_ip_var.set(ip)
    try:
        response: Response = await call_next(request)
    finally:
        client_ip_var.reset(token)

    process_time = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    if request.method == "GET" and response.status_code == 200 and request.url.path not in ("/", "/health", "/docs", "/openapi.json"):
        if not response.headers.get("Cache-Control"):
            response.headers["Cache-Control"] = "public, max-age=60, s-maxage=300, stale-while-revalidate=600"

    return response


app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SUBJECT_TYPE_MAP = {
    "all": SubjectType.ALL,
    "movies": SubjectType.MOVIES,
    "movie": SubjectType.MOVIES,
    "tv_series": SubjectType.TV_SERIES,
    "tv": SubjectType.TV_SERIES,
    "series": SubjectType.TV_SERIES,
    "anime": SubjectType.ANIME,
    "music": SubjectType.MUSIC,
    "education": SubjectType.EDUCATION,
}


def _map_subject_type(type_str: str) -> SubjectType:
    return _SUBJECT_TYPE_MAP.get((type_str or "all").lower().strip(), SubjectType.ALL)


_SUBJECT_TYPE_NAME = {
    SubjectType.ALL.value: "ALL",
    SubjectType.MOVIES.value: "MOVIES",
    SubjectType.TV_SERIES.value: "TV_SERIES",
    SubjectType.EDUCATION.value: "EDUCATION",
    SubjectType.MUSIC.value: "MUSIC",
    SubjectType.ANIME.value: "ANIME",
}


def _resolve_spoofed_ip(params: dict | None = None, json_body: dict | None = None) -> str | None:
    """Resolve regional spoofed IP based on host parameters or client request."""
    host = None
    if params and "host" in params:
        host = params["host"]
    elif json_body and "host" in json_body:
        host = json_body["host"]

    if host == "moviebox.com.bd":
        return "103.191.240.1"
    elif host == "moviebox.ph":
        return "112.198.115.36"

    ip = client_ip_var.get()

    if not ip or ip in ("127.0.0.1", "localhost", "::1"):
        return "103.191.240.1"

    if (
        ip.startswith("10.") or 
        ip.startswith("192.168.") or 
        ip.startswith("172.16.") or 
        ip.startswith("172.17.") or 
        ip.startswith("172.18.") or 
        ip.startswith("172.19.") or 
        ip.startswith("172.2") or 
        ip.startswith("172.3")
    ):
        return "103.191.240.1"

    return ip


async def _api_get(path: str, params: dict | None = None, retry: bool = True) -> dict | list:
    """GET an H5 endpoint with authorization token and unwrap the data envelope."""
    global _bearer_token
    token = await _get_bearer_token()
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {token}" if token else ""
    }
    ip = _resolve_spoofed_ip(params=params)
    if ip:
        headers.update({
            "X-Forwarded-For": ip,
            "X-Real-IP": ip,
            "Client-IP": ip,
            "CF-Connecting-IP": ip
        })

    client = getattr(app.state, "client", None)
    if client is None:
        async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
            r = await temp_client.get(BASE + path, params=params or {}, headers=headers)
    else:
        r = await client.get(BASE + path, params=params or {}, headers=headers)

    x_user = r.headers.get("x-user")
    if x_user:
        try:
            new_tok = json.loads(x_user).get("token")
            if new_tok:
                _bearer_token = new_tok
        except Exception:
            pass

    j = r.json()
    if j.get("code") == 400 and "token" in str(j.get("message", "")).lower() and retry:
        await _get_bearer_token(force_refresh=True)
        return await _api_get(path, params=params, retry=False)

    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j.get("data", {})
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


async def _rec_get(path: str, params: dict | None = None) -> dict | list:
    """GET an endpoint on the recommendation host (h5.aoneroom.com)."""
    token = await _get_bearer_token()
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {token}" if token else ""
    }
    ip = _resolve_spoofed_ip(params=params)
    if ip:
        headers.update({
            "X-Forwarded-For": ip,
            "X-Real-IP": ip,
            "Client-IP": ip,
            "CF-Connecting-IP": ip
        })

    client = getattr(app.state, "client", None)
    if client is None:
        async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
            r = await temp_client.get(REC_BASE + path, params=params or {}, headers=headers)
    else:
        r = await client.get(REC_BASE + path, params=params or {}, headers=headers)

    j = r.json()
    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j.get("data", {})
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


async def _api_post(path: str, json_body: dict, retry: bool = True) -> dict | list:
    """POST to an H5 endpoint with authorization token and unwrap data envelope."""
    global _bearer_token
    token = await _get_bearer_token()
    headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {token}" if token else ""
    }
    ip = _resolve_spoofed_ip(json_body=json_body)
    if ip:
        headers.update({
            "X-Forwarded-For": ip,
            "X-Real-IP": ip,
            "Client-IP": ip,
            "CF-Connecting-IP": ip
        })

    client = getattr(app.state, "client", None)
    if client is None:
        async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
            r = await temp_client.post(BASE + path, json=json_body, headers=headers)
    else:
        r = await client.post(BASE + path, json=json_body, headers=headers)

    x_user = r.headers.get("x-user")
    if x_user:
        try:
            new_tok = json.loads(x_user).get("token")
            if new_tok:
                _bearer_token = new_tok
        except Exception:
            pass

    j = r.json()
    if j.get("code") == 400 and "token" in str(j.get("message", "")).lower() and retry:
        await _get_bearer_token(force_refresh=True)
        return await _api_post(path, json_body=json_body, retry=False)

    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j.get("data", {})
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


# --- routes ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return JSONResponse({
            "status": "online",
            "service": "MovieBox API Pro (Ultra-Fast Edition)",
            "version": "2.2.0",
            "docs": "/docs",
            "endpoints": [
                "/home",
                "/homepage",
                "/movies",
                "/tv-series",
                "/animation",
                "/search?q=",
                "/search/suggest?q=",
                "/details/{slug_or_id}",
                "/download/{slug_or_id}?se=1&ep=1",
                "/api/stream/{subject_id}?detail_path=",
                "/api/stream/{subject_id}/captions?detail_path=",
                "/recommend/{slug_or_id}"
            ],
            "message": "High-Performance Pure REST API for moviebox.ph"
        })

    html_content = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MovieBox API Pro &bull; Next-Gen Streaming REST Gateway</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #06070a;
            --bg-surface: rgba(13, 16, 23, 0.7);
            --bg-surface-hover: rgba(22, 27, 39, 0.85);
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(99, 102, 241, 0.4);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.25);
            --accent-cyan: #06b6d4;
            --accent-pink: #ec4899;
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            overflow-x: hidden;
            background-image: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
                radial-gradient(circle at 90% 80%, rgba(6, 182, 212, 0.08), transparent 40%),
                radial-gradient(circle at 10% 90%, rgba(236, 72, 153, 0.08), transparent 40%);
            background-attachment: fixed;
        }
        .ambient-grid {
            position: fixed; inset: 0;
            background-image: linear-gradient(to right, rgba(255,255,255,0.02) 1px, transparent 1px),
                              linear-gradient(to bottom, rgba(255,255,255,0.02) 1px, transparent 1px);
            background-size: 40px 40px; pointer-events: none; z-index: 0;
        }
        .container { max-width: 1280px; margin: 0 auto; padding: 50px 24px 80px; position: relative; z-index: 1; }
        .navbar {
            display: flex; justify-content: space-between; align-items: center; padding: 16px 24px;
            background: rgba(13, 16, 23, 0.6); backdrop-filter: blur(16px);
            border: 1px solid var(--border); border-radius: 20px; margin-bottom: 50px;
        }
        .brand { display: flex; align-items: center; gap: 12px; text-decoration: none; color: var(--text-main); }
        .brand-icon {
            width: 38px; height: 38px; border-radius: 10px;
            background: linear-gradient(135deg, var(--primary), var(--accent-pink));
            display: flex; align-items: center; justify-content: center;
            box-shadow: 0 0 20px var(--primary-glow);
        }
        .brand-name { font-size: 1.25rem; font-weight: 800; letter-spacing: -0.5px; }
        .brand-tag {
            font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;
            background: rgba(99, 102, 241, 0.15); color: #818cf8; padding: 3px 8px;
            border-radius: 6px; border: 1px solid rgba(99, 102, 241, 0.3);
        }
        .nav-links { display: flex; align-items: center; gap: 12px; }
        .nav-btn {
            display: inline-flex; align-items: center; gap: 8px; padding: 8px 16px;
            font-size: 0.85rem; font-weight: 600; color: var(--text-muted); text-decoration: none;
            background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border);
            border-radius: 12px; transition: all 0.2s ease;
        }
        .nav-btn:hover { color: #fff; border-color: rgba(255, 255, 255, 0.2); background: rgba(255, 255, 255, 0.06); transform: translateY(-1px); }
        .hero { text-align: center; margin-bottom: 50px; }
        .status-pill {
            display: inline-flex; align-items: center; gap: 8px; padding: 6px 16px; border-radius: 30px;
            background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25);
            color: var(--accent-emerald); font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 20px;
        }
        .status-dot {
            width: 8px; height: 8px; border-radius: 50%; background: var(--accent-emerald);
            box-shadow: 0 0 10px var(--accent-emerald); animation: pulseDot 2s infinite ease-in-out;
        }
        @keyframes pulseDot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }
        .hero h1 {
            font-size: clamp(2.5rem, 6vw, 4.2rem); font-weight: 800; letter-spacing: -1.5px; line-height: 1.1; margin-bottom: 18px;
            background: linear-gradient(180deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero p { color: var(--text-muted); font-size: 1.15rem; max-width: 680px; margin: 0 auto 35px; line-height: 1.6; }
        .tester-box {
            max-width: 760px; margin: 0 auto 60px; background: var(--bg-surface); border: 1px solid var(--border);
            border-radius: 20px; padding: 8px; display: flex; gap: 8px; backdrop-filter: blur(12px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4); transition: border-color 0.3s;
        }
        .tester-box:focus-within { border-color: var(--primary); box-shadow: 0 0 30px var(--primary-glow); }
        .tester-input { flex-grow: 1; background: transparent; border: none; outline: none; padding: 12px 18px; font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; color: #fff; }
        .tester-input::placeholder { color: var(--text-dim); }
        .tester-btn {
            background: linear-gradient(135deg, var(--primary), #4f46e5); color: #fff; border: none;
            border-radius: 14px; padding: 12px 24px; font-weight: 700; font-size: 0.9rem; cursor: pointer;
            display: flex; align-items: center; gap: 8px; transition: all 0.2s;
        }
        .tester-btn:hover { opacity: 0.92; transform: translateY(-1px); box-shadow: 0 10px 20px var(--primary-glow); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 24px; }
        .card {
            background: var(--bg-surface); border: 1px solid var(--border); border-radius: 24px;
            padding: 30px; display: flex; flex-direction: column; backdrop-filter: blur(14px);
            position: relative; overflow: hidden; transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, transparent, var(--primary), transparent); opacity: 0; transition: opacity 0.3s; }
        .card:hover { background: var(--bg-surface-hover); border-color: var(--border-hover); transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0,0,0,0.5); }
        .card:hover::before { opacity: 1; }
        .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
        .card-icon-wrapper {
            width: 44px; height: 44px; border-radius: 14px; background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08); display: flex; align-items: center; justify-content: center;
            color: var(--accent-cyan); transition: all 0.3s;
        }
        .card:hover .card-icon-wrapper { background: rgba(99, 102, 241, 0.15); border-color: rgba(99, 102, 241, 0.3); color: #fff; transform: scale(1.05); }
        .method-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 700; padding: 4px 10px; border-radius: 8px; background: rgba(6, 182, 212, 0.1); color: var(--accent-cyan); border: 1px solid rgba(6, 182, 212, 0.2); text-transform: uppercase; }
        .card-title { font-size: 1.25rem; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 10px; }
        .card-desc { color: var(--text-muted); font-size: 0.92rem; line-height: 1.6; margin-bottom: 22px; flex-grow: 1; }
        .endpoint-pill { font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; background: rgba(0, 0, 0, 0.45); border: 1px solid rgba(255, 255, 255, 0.06); padding: 12px 14px; border-radius: 12px; color: #cbd5e1; display: flex; align-items: center; justify-content: space-between; margin-bottom: 18px; word-break: break-all; }
        .card-actions { display: flex; gap: 10px; }
        .action-btn { flex: 1; display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 12px; border-radius: 12px; font-size: 0.88rem; font-weight: 700; text-decoration: none; transition: all 0.2s; cursor: pointer; }
        .btn-launch { background: #ffffff; color: #000000; }
        .btn-launch:hover { background: var(--primary); color: #ffffff; box-shadow: 0 8px 20px var(--primary-glow); }
        .btn-copy { background: rgba(255, 255, 255, 0.04); color: var(--text-muted); border: 1px solid var(--border); max-width: 48px; padding: 12px; }
        .btn-copy:hover { background: rgba(255, 255, 255, 0.08); color: #fff; border-color: rgba(255, 255, 255, 0.2); }
        footer { margin-top: 80px; text-align: center; padding-top: 40px; border-top: 1px solid var(--border); color: var(--text-dim); font-size: 0.85rem; }
        .footer-tech { display: flex; justify-content: center; gap: 16px; margin-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-muted); }
        .toast {
            position: fixed; bottom: 30px; right: 30px; background: #1e293b; color: #fff; padding: 12px 20px;
            border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.4); font-size: 0.88rem; font-weight: 600;
            display: flex; align-items: center; gap: 10px; transform: translateY(100px); opacity: 0;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); z-index: 100;
        }
        .toast.show { transform: translateY(0); opacity: 1; }
        @media (max-width: 768px) {
            .container { padding: 30px 16px; }
            .grid { grid-template-columns: 1fr; }
            .navbar { flex-direction: column; gap: 16px; }
            .tester-box { flex-direction: column; }
        }
    </style>
</head>
<body>
    <div class="ambient-grid"></div>
    <div class="container">
        <nav class="navbar">
            <a href="/" class="brand">
                <div class="brand-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                </div>
                <div><span class="brand-name">MovieBox API</span></div>
                <span class="brand-tag">v2.2 Pro</span>
            </a>
            <div class="nav-links">
                <a href="/docs" target="_blank" class="nav-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Swagger UI
                </a>
                <a href="/redoc" target="_blank" class="nav-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
                    ReDoc
                </a>
                <a href="/health" target="_blank" class="nav-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--accent-emerald)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                    Health
                </a>
            </div>
        </nav>

        <header class="hero">
            <div class="status-pill"><div class="status-dot"></div>REST ENGINE ACTIVE &bull; HTTP/2 MULTIPLEXED</div>
            <h1>Ultra-Fast MovieBox<br>Streaming Gateway</h1>
            <p>Direct MP4 stream extractor, subtitle aggregator, real-time search engine, and metadata provider with zero web scraping.</p>
            <form class="tester-box" onsubmit="handleQuickTest(event)">
                <input id="quickInput" type="text" class="tester-input" placeholder="Search title or enter slug (e.g. Attack on Titan, Bad Sister)..." value="Attack on Titan">
                <button type="submit" class="tester-btn">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    Test Search
                </button>
            </form>
        </header>

        <div class="grid">
            <!-- 1. Universal Stream & Download Engine -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Unified Stream & Downloads</h3>
                <p class="card-desc">All direct MP4 streaming resolutions (360p - 1080p), file sizes in MB, and complete multi-language subtitles in one single call.</p>
                <div class="endpoint-pill"><span>/download/{slug_or_id}?se=1&ep=1</span></div>
                <div class="card-actions">
                    <a href="/download/attack-on-titan-hindi-kGWQOIx0d4?se=1&ep=1" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/download/attack-on-titan-hindi-kGWQOIx0d4?se=1&ep=1')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>

            <!-- 2. High-Precision Search -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Neural Search Engine</h3>
                <p class="card-desc">Instant indexing for full titles, posters, ratings, subject IDs, and detail slugs with automated guest Bearer JWT authorization.</p>
                <div class="endpoint-pill"><span>/search?q=Avengers</span></div>
                <div class="card-actions">
                    <a href="/search?q=Avengers" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/search?q=Avengers')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>

            <!-- 3. Search Autocomplete -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line><circle cx="12" cy="12" r="9"></circle></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Live Autocomplete Suggestions</h3>
                <p class="card-desc">Sub-millisecond query completions ideal for search bars, auto-fill inputs, and client-side drop-down preview widgets.</p>
                <div class="endpoint-pill"><span>/search/suggest?q=batman</span></div>
                <div class="card-actions">
                    <a href="/search/suggest?q=batman" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/search/suggest?q=batman')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>

            <!-- 4. Discover Home Feed -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Discover Home Feed</h3>
                <p class="card-desc">Real-time curated homepage blocks containing featured hero banners, top trending movies, serialized TV, and anime.</p>
                <div class="endpoint-pill"><span>/home</span></div>
                <div class="card-actions">
                    <a href="/home" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/home')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>

            <!-- 5. Metadata Details & Tree -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Metadata & Season Tree</h3>
                <p class="card-desc">Comprehensive item specs, episode lists, multi-language dub trees, actors, synopsis, and HD posters.</p>
                <div class="endpoint-pill"><span>/details/{slug_or_id}</span></div>
                <div class="card-actions">
                    <a href="/details/attack-on-titan-hindi-kGWQOIx0d4" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/details/attack-on-titan-hindi-kGWQOIx0d4')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>

            <!-- 6. Category Catalog Filters -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon-wrapper">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon></svg>
                    </div>
                    <span class="method-tag">GET</span>
                </div>
                <h3 class="card-title">Catalog Taxonomy Filters</h3>
                <p class="card-desc">Paginated library collections filtered by genre, release year, country, and language for Movies, Series, and Anime.</p>
                <div class="endpoint-pill"><span>/tv-series?page=1</span></div>
                <div class="card-actions">
                    <a href="/tv-series?page=1" target="_blank" class="action-btn btn-launch">Execute Query</a>
                    <button onclick="copyToClipboard('/tv-series?page=1')" class="action-btn btn-copy" title="Copy endpoint">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    </button>
                </div>
            </div>
        </div>

        <footer>
            <div>MovieBox Unofficial Pure REST Gateway &bull; Designed for High-Load Production</div>
            <div class="footer-tech">
                <span>FastAPI 0.110+</span>
                <span>&bull;</span>
                <span>HTTP/2 Multiplexing</span>
                <span>&bull;</span>
                <span>Gzip Compressed</span>
                <span>&bull;</span>
                <span>Zero Scraping</span>
            </div>
        </footer>
    </div>

    <div id="toast" class="toast">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent-emerald)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
        <span id="toastMsg">Endpoint copied to clipboard</span>
    </div>

    <script>
        function copyToClipboard(text) {
            const fullUrl = window.location.origin + text;
            navigator.clipboard.writeText(fullUrl).then(() => {
                showToast('Endpoint URL copied to clipboard!');
            }).catch(() => {
                showToast('Copied: ' + text);
            });
        }
        function showToast(msg) {
            const toast = document.getElementById('toast');
            document.getElementById('toastMsg').innerText = msg;
            toast.classList.add('show');
            setTimeout(() => { toast.classList.remove('show'); }, 2500);
        }
        function handleQuickTest(e) {
            e.preventDefault();
            const val = document.getElementById('quickInput').value.trim();
            if (!val) return;
            if (val.startsWith('/')) {
                window.open(val, '_blank');
            } else {
                window.open('/search?q=' + encodeURIComponent(val), '_blank');
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home")
async def get_home():
    """Formatted home feed with banners, movies, series, and animations."""
    cache_key = "home:formatted"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        data = await _api_get("/wefeed-h5api-bff/home", {"host": "moviebox.ph"})
        sections = []
        for op in (data or {}).get("operatingList", []) or []:
            op_type = op.get("type")
            title = op.get("title", "Featured")
            if op_type == "BANNER":
                items = [{
                    "name": item.get("title") or (item.get("subject") or {}).get("title"),
                    "poster_url": item.get("image", {}).get("url") or (item.get("subject") or {}).get("cover", {}).get("url"),
                    "slug": item.get("detailPath") or (item.get("subject") or {}).get("detailPath"),
                    "subject_id": (item.get("subject") or {}).get("subjectId"),
                    "badge": (item.get("subject") or {}).get("corner")
                } for item in op.get("banner", {}).get("items", []) if item.get("title") and "Communities" not in item.get("title")]
                sections.append({"section": "Banner", "count": len(items), "items": items})
            elif op_type in ["SUBJECTS_MOVIE", "SUBJECTS_TV", "SUBJECTS_ANIMATION"]:
                items = [{
                    "name": sub.get("title"),
                    "poster_url": (sub.get("cover") or {}).get("url"),
                    "slug": sub.get("detailPath"),
                    "subject_id": sub.get("subjectId"),
                    "badge": sub.get("corner"),
                    "rating": sub.get("imdbRatingValue")
                } for sub in op.get("subjects", [])]
                sections.append({"section": title, "count": len(items), "items": items})

        result = {"status": "success", "cached": False, "sections": sections}
        _cache_set(cache_key, result, HOMEPAGE_TTL)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching home: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/homepage")
async def get_homepage(
    host: str = Query(
        "moviebox.com.bd",
        description="MovieBox content host/region (e.g. moviebox.com.bd, moviebox.ph)",
    ),
):
    """Raw landing-page content listings (cached 5 min)."""
    cache_key = f"homepage:{host}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "cached": True, "host": host, "data": cached}
    try:
        data = await _api_get("/wefeed-h5api-bff/home", {"host": host})
        _cache_set(cache_key, data, HOMEPAGE_TTL)
        return {"status": "success", "cached": False, "host": host, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching homepage: {e}")
        raise HTTPException(status_code=502, detail=str(e))


async def _get_category_data(tab_id: int, page: int = 1, per_page: int = 24, sort: str = "RECOMMEND") -> dict:
    cache_key = f"category:{tab_id}:{page}:{per_page}:{sort}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    url = "/wefeed-h5api-bff/subject/filter"
    payload = {
        "tabId": tab_id,
        "filter": {"sort": sort, "genre": "ALL", "country": "ALL", "year": "ALL", "language": "ALL"},
        "page": page,
        "perPage": per_page
    }
    data = await _api_post(url, json_body=payload)
    raw_items = (data or {}).get("items", (data or {}).get("subjects", [])) or []
    items = [{
        "name": sub.get("title"),
        "poster_url": (sub.get("cover") or {}).get("url"),
        "slug": sub.get("detailPath"),
        "subject_id": sub.get("subjectId"),
        "badge": sub.get("corner"),
        "rating": sub.get("imdbRatingValue"),
        "year": sub.get("releaseDate", "")[:4] if sub.get("releaseDate") else None
    } for sub in raw_items]
    pager = (data or {}).get("pager", {}) or {}
    total = pager.get("totalCount") or (data or {}).get("total") or len(items)
    res = {"status": "success", "cached": False, "page": page, "per_page": per_page, "total": total, "items": items}
    _cache_set(cache_key, res, HOMEPAGE_TTL)
    return res


@app.get("/movies")
async def get_movies(page: int = Query(1, ge=1), sort: str = "RECOMMEND"):
    """Browse catalog movies with pagination."""
    return await _get_category_data(tab_id=2, page=page, sort=sort)


@app.get("/tv-series")
async def get_tv_series(page: int = Query(1, ge=1), sort: str = "RECOMMEND"):
    """Browse catalog TV series with pagination."""
    return await _get_category_data(tab_id=5, page=page, sort=sort)


@app.get("/animation")
async def get_animation(page: int = Query(1, ge=1), sort: str = "RECOMMEND"):
    """Browse catalog anime / animation with pagination."""
    return await _get_category_data(tab_id=8, page=page, sort=sort)


@app.get("/search/suggest")
async def get_search_suggestions(q: str = Query(..., min_length=1, description="Keyword")):
    """Instant search suggestions / autocomplete."""
    cache_key = f"suggest:{q.lower().strip()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "cached": True, **cached}

    try:
        data = await _api_post(
            "/wefeed-h5api-bff/subject/search-suggest",
            json_body={"keyword": q, "perPage": 10}
        )
        raw = (data or {}).get("items", (data or {}).get("list", [])) or []
        suggestions = []
        for item in raw:
            sub = item.get("subject") or {}
            suggestions.append({
                "title": sub.get("title") or item.get("word") or item.get("title"),
                "slug": sub.get("detailPath") or item.get("detailPath"),
                "subject_id": sub.get("subjectId") or item.get("subjectId")
            })
        payload = {"query": q, "count": len(suggestions), "suggestions": suggestions}
        _cache_set(cache_key, payload, SEARCH_TTL)
        return {"status": "success", "cached": False, **payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching search suggestions for '{q}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1, description="Search keyword"),
    type: str = Query(
        "all",
        description="Content type (all, movies, tv_series, anime, music, education)",
    ),
    page: int = Query(1, ge=1, description="Page number"),
):
    """Search movies / tv-series / anime with high accuracy (cached 2 min)."""
    subject_type = _map_subject_type(type)
    cache_key = f"search:{subject_type.value}:{page}:{q.lower().strip()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "cached": True, **cached}

    try:
        data = await _api_post(
            "/wefeed-h5api-bff/subject/search",
            json_body={"keyword": q, "page": page, "perPage": 24},
        )
        raw_items = (data or {}).get("items", (data or {}).get("list", [])) or []
        
        items = [{
            "name": sub.get("title"),
            "poster_url": (sub.get("cover") or {}).get("url"),
            "slug": sub.get("detailPath"),
            "subject_id": sub.get("subjectId"),
            "subject_type": sub.get("subjectType"),
            "rating": sub.get("imdbRatingValue"),
            "badge": sub.get("corner")
        } for sub in raw_items]

        if subject_type is not SubjectType.ALL:
            items = [it for it in items if it.get("subject_type") == subject_type.value]
            raw_items = [it for it in raw_items if it.get("subjectType") == subject_type.value]
            if isinstance(data, dict):
                data["items"] = raw_items

        pager = (data or {}).get("pager", {}) or {}
        total = pager.get("totalCount") or (data or {}).get("total") or len(items)

        payload = {
            "query": q,
            "type": type,
            "page": page,
            "total": total,
            "items": items,
            "data": data
        }
        _cache_set(cache_key, payload, SEARCH_TTL)
        return {"status": "success", "cached": False, **payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching '{q}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


def _shape_seasons(details_data: dict) -> dict:
    """Build a clean season/episode summary from a raw /detail payload."""
    data = details_data or {}
    subject = data.get("subject") or {}
    is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
    raw_seasons = ((data.get("resource") or {}).get("seasons")) or []

    seasons = []
    for s in raw_seasons:
        resolutions = s.get("resolutions") or []
        ep_from_res = max((r.get("epNum", 0) or 0 for r in resolutions), default=0)
        episode_count = ep_from_res or int(s.get("maxEp", 0) or 0)
        seasons.append(
            {
                "season": s.get("se"),
                "episode_count": episode_count,
                "resolutions": sorted(
                    {f"{r.get('resolution')}p" for r in resolutions if r.get("resolution")},
                    key=lambda x: int(x[:-1]),
                ),
            }
        )

    seasons.sort(key=lambda x: x.get("season") or 0)
    return {
        "is_series": is_series,
        "title": subject.get("title", "Unknown"),
        "season_count": len(seasons),
        "total_episodes": sum(s["episode_count"] for s in seasons),
        "seasons": seasons,
    }


def _shape_dubs(details_data: dict) -> list[dict]:
    """Build a clean list of available dubs/audio tracks from raw /detail payload."""
    data = details_data or {}
    subject = data.get("subject") or {}
    raw_dubs = subject.get("dubs") or []
    dubs = []
    for d in raw_dubs:
        dubs.append({
            "subject_id": str(d.get("subjectId")),
            "detail_path": d.get("detailPath"),
            "language_name": d.get("lanName"),
            "language_code": d.get("lanCode"),
            "is_original": d.get("original", False)
        })
    return dubs


TMDB_API_KEY = os.getenv("TMDB_API_KEY", "3356865d41894a2fa9bfa84b2b5f59bb")


def _clean_title(title: str) -> str:
    if not title:
        return ""
    t = re.sub(r"\[.*?\]|\(.*?\)", "", title)
    t = re.sub(r":\s*Season\s*\d+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"Season\s*\d+", "", t, flags=re.IGNORECASE)
    return t.strip()


async def _resolve_tmdb_info(title: str, year: str = "", is_series: bool | None = None) -> dict:
    """Ultra-fast cached TMDB ID, Logo, Backdrop & Poster resolver."""
    cleaned = _clean_title(title)
    if not cleaned:
        return {"tmdb_id": None, "logo": None, "logo_w500": None, "backdrop": None, "poster": None, "logos": []}

    media_type_key = "tv" if is_series is True else ("movie" if is_series is False else "multi")
    year_str = str(year)[:4] if year else ""
    cache_key = f"tmdb_meta:{media_type_key}:{cleaned}:{year_str}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    client = getattr(app.state, "client", None)
    should_close = False
    if client is None:
        client = httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True)
        should_close = True

    tmdb_id = None
    poster_path = None
    backdrop_path = None
    logo_url = None
    logo_w500 = None
    all_logos = []
    resolved_media_type = "tv" if is_series is True else "movie"

    try:
        # Fast TMDB Search (search multi or specific type)
        endpoint = "multi" if is_series is None else ("tv" if is_series else "movie")
        search_url = f"https://api.themoviedb.org/3/search/{endpoint}?api_key={TMDB_API_KEY}&query={quote(cleaned)}"
        r = await client.get(search_url, timeout=3.5)
        if r.status_code == 200:
            results = r.json().get("results", [])
            if is_series is not None and endpoint == "multi":
                filtered = [res for res in results if res.get("media_type") == ("tv" if is_series else "movie")]
                if filtered:
                    results = filtered
            if results:
                top = results[0]
                tmdb_id = top.get("id")
                poster_path = top.get("poster_path")
                backdrop_path = top.get("backdrop_path")
                if "media_type" in top:
                    resolved_media_type = top.get("media_type")

        # Fetch title logos from TMDB images if TMDB ID resolved
        if tmdb_id:
            try:
                img_url = f"https://api.themoviedb.org/3/{resolved_media_type}/{tmdb_id}/images?api_key={TMDB_API_KEY}"
                r_img = await client.get(img_url, timeout=3.5)
                if r_img.status_code == 200:
                    img_data = r_img.json()
                    raw_logos = img_data.get("logos", [])
                    for l in raw_logos:
                        fp = l.get("file_path")
                        if fp:
                            full_p = f"https://image.tmdb.org/t/p/original{fp}"
                            all_logos.append({
                                "url": full_p,
                                "url_w500": f"https://image.tmdb.org/t/p/w500{fp}",
                                "aspect_ratio": l.get("aspect_ratio"),
                                "width": l.get("width"),
                                "height": l.get("height"),
                                "lang": l.get("iso_639_1"),
                                "vote_average": l.get("vote_average"),
                                "vote_count": l.get("vote_count"),
                            })
                    # Smart English-First Logo Selection
                    NON_LATIN_LANGS = {"zh", "ja", "ko", "ar", "ru", "hi", "th", "he", "fa", "el", "ta", "te", "bn"}
                    en_logos = [l for l in all_logos if l.get("lang") == "en"]
                    picked_logo = None

                    if en_logos:
                        best_en = sorted(en_logos, key=lambda x: (x.get("vote_average", 0), x.get("vote_count", 0)), reverse=True)[0]
                        top_global = all_logos[0]
                        global_lang = top_global.get("lang")
                        # If top rated global is in latin script and has significantly higher community votes, use it
                        if global_lang not in NON_LATIN_LANGS and top_global.get("vote_average", 0) > best_en.get("vote_average", 0) + 1.0:
                            picked_logo = top_global
                        else:
                            picked_logo = best_en
                    elif all_logos:
                        latin_logos = [l for l in all_logos if l.get("lang") not in NON_LATIN_LANGS]
                        picked_logo = latin_logos[0] if latin_logos else all_logos[0]

                    if picked_logo:
                        logo_url = picked_logo["url"]
                        logo_w500 = picked_logo["url_w500"]
            except Exception:
                pass
    finally:
        if should_close:
            await client.aclose()

    res = {
        "tmdb_id": tmdb_id,
        "media_type": resolved_media_type,
        "logo": logo_url,
        "logo_w500": logo_w500,
        "backdrop": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None,
        "poster": f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None,
        "logos": all_logos
    }
    _cache_set(cache_key, res, METADATA_TTL)
    return res


async def _fetch_net27_stream_sources(
    tmdb_id: int,
    is_series: bool,
    se: int = 1,
    ep: int = 1,
    subject_id: str = "",
    detail_path: str = "",
    dubs: list | None = None
) -> dict | None:
    """Fetch all high-resolution direct CDN MP4 streams and captions from Net27 engine."""
    if not tmdb_id:
        return None

    media_type = "tv" if is_series else "movie"
    url = f"https://net27.cc/api/embed-tmdb/{tmdb_id}?type={media_type}"
    if is_series:
        url += f"&se={se}&ep={ep}"
    if subject_id:
        url += f"&sid={subject_id}"
    if detail_path:
        url += f"&dp={detail_path}"

    if dubs:
        warm_parts = []
        for d in dubs:
            sid = d.get("subjectId") or d.get("id") or d.get("subject_id")
            dp = d.get("detailPath") or d.get("detail_path")
            if sid and dp:
                warm_parts.append(f"{sid}~{dp}")
        if warm_parts:
            url += f"&warm={quote(','.join(warm_parts))}"

    client = getattr(app.state, "client", None)
    should_close = False
    if client is None:
        client = httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True)
        should_close = True

    try:
        resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("ok") and data.get("streams"):
                return data
    except Exception as e:
        logger.warning(f"Net27 stream fetch notice: {e}")
    finally:
        if should_close:
            await client.aclose()
    return None


async def _fetch_details(detail_path: str) -> dict:
    """Fetch + cache raw item details via the shared client."""
    is_numeric = str(detail_path).isdigit()
    cache_key = f"details:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    params = {"subjectId": detail_path} if is_numeric else {"detailPath": detail_path}
    data = await _api_get("/wefeed-h5api-bff/detail", params)
    
    # Store under both slug and ID for instant unified resolution
    subj = (data or {}).get("subject") or {}
    subj_id = subj.get("subjectId")
    slug = subj.get("detailPath")
    if subj_id:
        _cache_set(f"details:{subj_id}", data, DETAILS_TTL)
    if slug:
        _cache_set(f"details:{slug}", data, DETAILS_TTL)

    _cache_set(cache_key, data, DETAILS_TTL)
    return data


def _extract_slug_title(slug: str) -> str:
    """Fast candidate title extractor from Moviebox URL slug."""
    if not slug or str(slug).isdigit():
        return ""
    parts = slug.split("-")
    if len(parts) > 1 and len(parts[-1]) >= 8:
        parts = parts[:-1]
    filtered = [p for p in parts if p.lower() not in ("english", "hindi", "tamil", "telugu", "spanish", "french", "season", "complete", "dub", "sub", "hd", "4k")]
    return " ".join(filtered) if filtered else " ".join(parts)


async def _background_prefetch_next_ep(detail_path: str, season: int, episode: int):
    """Fire-and-forget background pre-warm for next episode into cache."""
    try:
        await get_download_links(detail_path=detail_path, season=season, episode=episode)
    except Exception:
        pass


@app.get("/details/{detail_path}")
@app.get("/detail/{detail_path}")
async def get_details(detail_path: str):
    """Specific item details with TMDB ID & Title Logo (cached 10 min)."""
    cache_key = f"details_resp:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}
    try:
        candidate = _extract_slug_title(detail_path)
        
        # Parallel Step: Fetch MovieBox Detail + TMDB Info simultaneously
        if candidate:
            task_detail = _fetch_details(detail_path)
            task_tmdb = _resolve_tmdb_info(candidate)
            data, tmdb_info = await asyncio.gather(task_detail, task_tmdb)
        else:
            data = await _fetch_details(detail_path)
            subject = (data or {}).get("subject") or {}
            title = subject.get("title", "")
            year = subject.get("releaseDate", "")
            is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
            tmdb_info = await _resolve_tmdb_info(title, year, is_series)

        result = {
            "status": "success",
            "cached": False,
            "detail_path": detail_path,
            "tmdb_id": tmdb_info.get("tmdb_id"),
            "logo": tmdb_info.get("logo"),
            "logo_w500": tmdb_info.get("logo_w500"),
            "backdrop": tmdb_info.get("backdrop"),
            "poster": tmdb_info.get("poster"),
            "logos": tmdb_info.get("logos", []),
            "seasons": _shape_seasons(data),
            "dubs": _shape_dubs(data),
            "data": data,
        }
        _cache_set(cache_key, result, DETAILS_TTL)
        
        # Predictive background stream pre-warm: Fetch streams while user is reading details
        asyncio.create_task(_background_prefetch_next_ep(detail_path, 1, 1))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching details for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/download/{detail_path}")
async def get_download_links(
    detail_path: str,
    season: int = 0,
    episode: int = 0,
    se: int = 0,
    ep: int = 0,
):
    """All available stream / download links + subtitles + logo in a single request.
    
    Uses high-speed Net27 engine (1080p/720p/480p/360p direct CDN MP4s) with Netfilm fallback.
    """
    season_val = max(int(season or 0), int(se or 0))
    episode_val = max(int(episode or 0), int(ep or 0))

    try:
        candidate = _extract_slug_title(detail_path)
        
        # Parallel Step 1: Fetch MovieBox Detail + TMDB Info concurrently
        hint_series = True if (season_val > 0 or episode_val > 0) else None
        if candidate:
            task_detail = _fetch_details(detail_path)
            task_tmdb = _resolve_tmdb_info(candidate, is_series=hint_series)
            details_data, tmdb_info = await asyncio.gather(task_detail, task_tmdb)
        else:
            details_data = await _fetch_details(detail_path)
            subject = (details_data or {}).get("subject") or {}
            title = subject.get("title", "Unknown")
            year = subject.get("releaseDate", "")
            is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
            tmdb_info = await _resolve_tmdb_info(title, year, is_series)

        subject = (details_data or {}).get("subject") or {}
        subject_id = str(subject.get("subjectId") or detail_path)
        detail_path_slug = str(subject.get("detailPath") or detail_path)
        title = subject.get("title", "Unknown")
        year = subject.get("releaseDate", "")
        raw_dubs = subject.get("dubs", [])

        is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
        eff_se = season_val
        eff_ep = episode_val
        if is_series and season_val == 0 and episode_val == 0:
            eff_se = 1
            eff_ep = 1

        # Check unified cache key by subject_id
        cache_key = f"download:{subject_id}:{eff_se}:{eff_ep}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        # Verify TMDB match against exact subjectType
        resolved_type = tmdb_info.get("media_type")
        expected_type = "tv" if is_series else "movie"
        if not tmdb_info.get("tmdb_id") or (resolved_type and resolved_type != expected_type):
            tmdb_info = await _resolve_tmdb_info(title, year, is_series)

        tmdb_id = tmdb_info.get("tmdb_id")

        files = []
        captions = []
        hls = []
        dash = []
        has_res = False

        # 2. Try Primary Net27 Engine for full resolutions (360p, 480p, 720p, 1080p)
        if tmdb_id:
            net27_data = await _fetch_net27_stream_sources(
                tmdb_id=tmdb_id,
                is_series=is_series,
                se=eff_se,
                ep=eff_ep,
                subject_id=subject_id,
                detail_path=detail_path_slug,
                dubs=raw_dubs
            )
            if net27_data and net27_data.get("streams"):
                has_res = True
                for s in net27_data.get("streams", []):
                    url = s.get("url") or ""
                    if not url:
                        continue
                    res_val = s.get("resolution")
                    size_b = int(s.get("size", 0) or 0)
                    files.append({
                        "resolution": f"{res_val}p" if res_val else "unknown",
                        "resolution_value": int(res_val) if str(res_val).isdigit() else 0,
                        "size_bytes": size_b,
                        "size_mb": round(size_b / (1024 * 1024), 2),
                        "ext": "mp4",
                        "id": str(s.get("id", "")),
                        "stream_link": url,
                        "codec": s.get("codec") or "h264",
                        "vip_locked": False
                    })
                # Sort files by resolution descending (1080p, 720p, 480p, 360p)
                files.sort(key=lambda x: x.get("resolution_value", 0), reverse=True)

                # Parse captions from Net27
                for c in net27_data.get("captions", []):
                    c_url = c.get("url") or ""
                    if "url=" in c_url:
                        c_url = c_url.split("url=", 1)[-1]
                        c_url = unquote(c_url)
                    captions.append({
                        "language": c.get("name") or c.get("lang"),
                        "language_code": c.get("lang"),
                        "size_bytes": 0,
                        "delay": 0,
                        "url": c_url,
                    })

                if net27_data.get("fallbackHls"):
                    hls.append({"url": net27_data.get("fallbackHls"), "format": "HLS"})

        # 3. Fallback to Legacy Netfilm/Moviebox Engine if Net27 has no files
        if not files:
            dom_data = await _api_get("/wefeed-h5api-bff/media-player/get-domain")
            domain = str(dom_data if isinstance(dom_data, str) else (dom_data.get("data") if isinstance(dom_data, dict) else "https://netfilm.world")).rstrip("/")
            if not domain.startswith("http"):
                domain = "https://netfilm.world"

            type_str = "/tv/detail" if is_series else "/movie/detail"
            player_referer = (
                f"{domain}/spa/videoPlayPage/movies/{detail_path_slug}"
                f"?id={subject_id}&type={type_str}&detailSe={eff_se}&detailEp={eff_ep}&lang=en"
            )
            play_url = f"{domain}/wefeed-h5api-bff/subject/play?subjectId={subject_id}&se={eff_se}&ep={eff_ep}&detailPath={detail_path_slug}"

            token = await _get_bearer_token()
            ip = _resolve_spoofed_ip()
            ip_headers = {}
            if ip:
                ip_headers = {
                    "X-Forwarded-For": ip,
                    "X-Real-IP": ip,
                    "Client-IP": ip,
                    "CF-Connecting-IP": ip,
                }

            headers = {
                **PLAYER_HEADERS,
                **ip_headers,
                "Referer": player_referer,
                "Authorization": f"Bearer {token}" if token else ""
            }

            client = getattr(app.state, "client", None)
            if client is None:
                async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
                    play_resp = await temp_client.get(play_url, headers=headers)
            else:
                play_resp = await client.get(play_url, headers=headers)

            play_data = play_resp.json().get("data", {}) if play_resp.status_code == 200 else {}
            streams = play_data.get("streams", [])
            dash = play_data.get("dash", [])
            hls = play_data.get("hls", [])

            if not streams and not dash and detail_path_slug != subject_id:
                try:
                    retry_url = f"{domain}/wefeed-h5api-bff/subject/play?subjectId={subject_id}&se={eff_se}&ep={eff_ep}&detailPath={subject_id}"
                    if client is None:
                        async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
                            retry_resp = await temp_client.get(retry_url, headers=headers)
                    else:
                        retry_resp = await client.get(retry_url, headers=headers)
                    retry_data = retry_resp.json().get("data", {}) if retry_resp.status_code == 200 else {}
                    if retry_data.get("streams"):
                        play_data = retry_data
                        streams = play_data.get("streams", [])
                        dash = play_data.get("dash", [])
                        hls = play_data.get("hls", [])
                except Exception:
                    pass

            for s in streams:
                url = s.get("url") or ""
                if not url:
                    continue
                res_val = s.get("resolutions")
                size_b = int(s.get("size", 0) or 0)
                res_str = f"{res_val}p" if res_val else "unknown"
                files.append({
                    "resolution": res_str,
                    "resolution_value": int(res_val) if str(res_val).isdigit() else 0,
                    "size_bytes": size_b,
                    "size_mb": round(size_b / (1024 * 1024), 2),
                    "ext": "mp4",
                    "id": str(s.get("id", "")),
                    "stream_link": url,
                    "codec": s.get("codecName"),
                    "duration": s.get("duration"),
                    "vip_locked": s.get("vipLocked", False)
                })

            # Fetch captions for legacy stream
            stream_id = None
            stream_format = "MP4"
            if streams:
                stream_id = streams[0].get("id")
                stream_format = streams[0].get("format", "MP4")
            elif dash:
                stream_id = dash[0].get("id")
                stream_format = dash[0].get("format", "DASH")

            if stream_id and not captions:
                try:
                    cap_params = {
                        "format": stream_format,
                        "id": str(stream_id),
                        "subjectId": str(subject_id),
                        "detailPath": str(detail_path_slug)
                    }
                    cap_data = await _api_get("/wefeed-h5api-bff/subject/caption", params=cap_params)
                    raw_caps = cap_data.get("captions", []) if isinstance(cap_data, dict) else (cap_data if isinstance(cap_data, list) else [])
                    for c in raw_caps or []:
                        captions.append({
                            "language": c.get("lanName") or c.get("lan"),
                            "language_code": c.get("lan"),
                            "size_bytes": int(c.get("size", 0) or 0),
                            "delay": c.get("delay", 0),
                            "url": c.get("url"),
                        })
                except Exception as e:
                    logger.warning(f"Failed to fetch captions: {e}")

        cover = subject.get("cover") or {}
        subject_type = subject.get("subjectType")
        valid_stream_files = [f for f in files if f.get("stream_link")]
        has_res = len(valid_stream_files) > 0 or has_res

        result = {
            "status": "success",
            "detail_path": detail_path_slug,
            "subject_id": str(subject_id),
            "tmdb_id": tmdb_info.get("tmdb_id"),
            "logo": tmdb_info.get("logo"),
            "logo_w500": tmdb_info.get("logo_w500"),
            "backdrop": tmdb_info.get("backdrop"),
            "poster": tmdb_info.get("poster"),
            "title": title,
            "subject_type": _SUBJECT_TYPE_NAME.get(subject_type, str(subject_type)),
            "season": eff_se if is_series else None,
            "episode": eff_ep if is_series else None,
            "release_date": subject.get("releaseDate", ""),
            "cover_image": cover.get("url"),
            "has_resource": has_res,
            "limited": False,
            "qualities_count": len(files),
            "files": files,
            "subtitles": captions,
            "hls": hls,
            "dash": dash,
            "cached": False
        }
        
        # Cache when files are extracted
        if len(valid_stream_files) > 0:
            _cache_set(cache_key, result, DETAILS_TTL)
            if detail_path != subject_id:
                _cache_set(f"download:{detail_path}:{eff_se}:{eff_ep}", result, DETAILS_TTL)
            # Predictive background pre-warm for next episode
            if is_series and eff_ep > 0:
                asyncio.create_task(_background_prefetch_next_ep(detail_path_slug, eff_se, eff_ep + 1))

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching download links for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stream/{subject_id}")
async def get_stream_sources(subject_id: str, detail_path: str = "", se: int = 1, ep: int = 1):
    """Direct stream engine integration with Net27 1080p/720p/480p/360p + legacy fallback."""
    try:
        slug = detail_path or subject_id
        dl_res = await get_download_links(detail_path=slug, season=se, episode=ep)
        
        sources = [
            {
                "resolution": f.get("resolution"),
                "format": f.get("ext", "mp4").upper(),
                "url": f.get("stream_link"),
                "size": f.get("size_bytes"),
                "duration": f.get("duration"),
                "codec": f.get("codec")
            }
            for f in dl_res.get("files", [])
            if f.get("stream_link")
        ]

        return {
            "status": "success",
            "subject_id": str(dl_res.get("subject_id") or subject_id),
            "detail_path": dl_res.get("detail_path") or detail_path,
            "tmdb_id": dl_res.get("tmdb_id"),
            "logo": dl_res.get("logo"),
            "logo_w500": dl_res.get("logo_w500"),
            "backdrop": dl_res.get("backdrop"),
            "poster": dl_res.get("poster"),
            "se": se,
            "ep": ep,
            "has_resource": dl_res.get("has_resource", False),
            "sources": sources,
            "hls": dl_res.get("hls", []),
            "dash": dl_res.get("dash", []),
            "free_episodes": 999,
            "limited": False,
            "note": None if dl_res.get("has_resource") else "No stream found for this episode."
        }
    except Exception as e:
        logger.error(f"Error in stream extraction: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stream/{subject_id}/captions")
async def get_captions(subject_id: str, detail_path: str = "", se: int = 1, ep: int = 1):
    """Fetch captions for a stream subject with Net27 + legacy fallback."""
    try:
        slug = detail_path or subject_id
        dl_res = await get_download_links(detail_path=slug, season=se, episode=ep)
        captions = dl_res.get("subtitles", [])
        return {
            "status": "success",
            "subject_id": subject_id,
            "se": se,
            "ep": ep,
            "count": len(captions),
            "captions": captions
        }
    except Exception as e:
        logger.error(f"Error fetching captions: {e}")
        raise HTTPException(status_code=502, detail=str(e))


def _shape_recommend_item(it: dict) -> dict:
    """Trim a raw recommend item to the useful fields."""
    cover = it.get("cover") or {}
    stype = it.get("subjectType")
    return {
        "title": it.get("title"),
        "subject_id": it.get("subjectId"),
        "detail_path": it.get("detailPath"),
        "subject_type": _SUBJECT_TYPE_NAME.get(stype, str(stype)),
        "release_date": it.get("releaseDate", ""),
        "genre": it.get("genre", ""),
        "imdb_rating": it.get("imdbRatingValue") or it.get("imdbRate"),
        "cover_image": cover.get("url"),
    }


@app.get("/recommend/{detail_path}")
async def get_recommendations(
    detail_path: str,
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(12, ge=1, le=48, description="Items per page"),
):
    """"More like this" - related movies / series for a given item."""
    cache_key = f"recommend:{detail_path}:{page}:{per_page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        subject_id = detail_path
        if not str(detail_path).isdigit():
            details_data = await _fetch_details(detail_path)
            subject_id = ((details_data or {}).get("subject") or {}).get("subjectId")
            if not subject_id:
                raise HTTPException(status_code=404, detail="Could not resolve subjectId")

        data = await _rec_get(
            "/wefeed-h5-bff/web/subject/detail-rec",
            {"subjectId": subject_id, "page": page, "perPage": per_page},
        )
        items = (data or {}).get("items", []) or []
        shaped = [_shape_recommend_item(it) for it in items]
        result = {
            "status": "success",
            "cached": False,
            "detail_path": detail_path,
            "subject_id": subject_id,
            "page": page,
            "count": len(shaped),
            "items": shaped,
        }
        _cache_set(cache_key, result, RECOMMEND_TTL)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendations for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


# ============================================================================
# TMDB-POWERED NETFLIX-GRADE HOMEPAGE & DIRECT STREAMING ENGINE
# ============================================================================

HOMEPAGE_TTL = 1800.0  # 30 Minutes


def _format_tmdb_card(item: dict, default_type: str = "movie") -> dict:
    """Format a raw TMDB result into a clean, unified movie/series card."""
    media_type = item.get("media_type") or default_type
    is_tv = media_type == "tv"
    title = item.get("title") or item.get("name") or "Unknown"
    orig_title = item.get("original_title") or item.get("original_name")
    poster_path = item.get("poster_path")
    backdrop_path = item.get("backdrop_path")
    vote_avg = item.get("vote_average", 0.0)
    tmdb_id = item.get("id")

    return {
        "id": tmdb_id,
        "tmdb_id": tmdb_id,
        "title": title,
        "original_title": orig_title,
        "media_type": media_type,
        "overview": item.get("overview", ""),
        "poster": f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None,
        "poster_w500": f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None,
        "backdrop": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None,
        "backdrop_w780": f"https://image.tmdb.org/t/p/w780{backdrop_path}" if backdrop_path else None,
        "rating": round(vote_avg, 1) if vote_avg else None,
        "vote_count": item.get("vote_count", 0),
        "release_date": item.get("release_date") or item.get("first_air_date") or "",
        "genre_ids": item.get("genre_ids", []),
        "popularity": item.get("popularity", 0.0),
        "detail_url": f"/details/tmdb/{tmdb_id}?type={media_type}",
        "stream_url": f"/download/tmdb/{tmdb_id}?type={media_type}"
    }


@app.get("/api/home")
@app.get("/homepage")
@app.get("/tmdb/home")
async def get_homepage():
    """Ultra-fast, Netflix-grade homepage powered directly by TMDB real-time data."""
    cache_key = "tmdb_homepage_payload"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    client = getattr(app.state, "client", None)
    should_close = False
    if client is None:
        client = httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True)
        should_close = True

    try:
        # Concurrent parallel requests for all homepage sections
        t_trending_day = client.get(f"https://api.themoviedb.org/3/trending/all/day?api_key={TMDB_API_KEY}", timeout=5.0)
        t_mov_week = client.get(f"https://api.themoviedb.org/3/trending/movie/week?api_key={TMDB_API_KEY}", timeout=5.0)
        t_tv_week = client.get(f"https://api.themoviedb.org/3/trending/tv/week?api_key={TMDB_API_KEY}", timeout=5.0)
        t_anime = client.get(f"https://api.themoviedb.org/3/discover/tv?api_key={TMDB_API_KEY}&with_genres=16&sort_by=popularity.desc", timeout=5.0)
        t_kdrama = client.get(f"https://api.themoviedb.org/3/discover/tv?api_key={TMDB_API_KEY}&with_original_language=ko&sort_by=popularity.desc", timeout=5.0)
        t_action = client.get(f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_genres=28&sort_by=popularity.desc", timeout=5.0)
        t_top_rated = client.get(f"https://api.themoviedb.org/3/movie/top_rated?api_key={TMDB_API_KEY}", timeout=5.0)

        r_trend, r_mov, r_tv, r_anime, r_kdrama, r_act, r_top = await asyncio.gather(
            t_trending_day, t_mov_week, t_tv_week, t_anime, t_kdrama, t_action, t_top_rated,
            return_exceptions=True
        )

        def _safe_results(r, def_type="movie"):
            if isinstance(r, httpx.Response) and r.status_code == 200:
                raw_list = r.json().get("results", [])
                return [_format_tmdb_card(it, def_type) for it in raw_list]
            return []

        trending_all = _safe_results(r_trend, "movie")
        trending_movies = _safe_results(r_mov, "movie")
        trending_tv = _safe_results(r_tv, "tv")
        anime_list = _safe_results(r_anime, "tv")
        kdrama_list = _safe_results(r_kdrama, "tv")
        action_list = _safe_results(r_act, "movie")
        top_rated_list = _safe_results(r_top, "movie")

        # Concurrently enrich Top 6 Hero Banner items with English transparent title logos
        hero_candidates = trending_all[:6]
        hero_tasks = [
            _resolve_tmdb_info(
                title=item.get("title", ""),
                year=item.get("release_date", ""),
                is_series=(item.get("media_type") == "tv")
            )
            for item in hero_candidates
        ]
        resolved_logos = await asyncio.gather(*hero_tasks, return_exceptions=True)

        hero_banner = []
        for i, item in enumerate(hero_candidates):
            logo_url = None
            logo_w500 = None
            if i < len(resolved_logos) and isinstance(resolved_logos[i], dict):
                logo_url = resolved_logos[i].get("logo")
                logo_w500 = resolved_logos[i].get("logo_w500")
            hero_banner.append({
                **item,
                "logo": logo_url,
                "logo_w500": logo_w500,
            })

        sections = [
            {"id": "trending_movies", "title": "🔥 Trending Movies", "items": trending_movies},
            {"id": "trending_tv", "title": "📺 Popular TV Shows", "items": trending_tv},
            {"id": "top_anime", "title": "⚔️ Top Rated Anime", "items": anime_list},
            {"id": "k_dramas", "title": "🇰🇷 Popular K-Dramas", "items": kdrama_list},
            {"id": "action_blockbusters", "title": "🍿 Action Blockbusters", "items": action_list},
            {"id": "top_rated", "title": "⭐ Top Rated Masterpieces", "items": top_rated_list},
        ]

        result = {
            "status": "success",
            "cached": False,
            "hero_banner": hero_banner,
            "sections": sections,
        }
        _cache_set(cache_key, result, HOMEPAGE_TTL)
        return result
    finally:
        if should_close:
            await client.aclose()


@app.get("/details/tmdb/{tmdb_id}")
@app.get("/detail/tmdb/{tmdb_id}")
async def get_tmdb_direct_details(
    tmdb_id: int,
    type: str = Query("movie", description="Media type: 'movie' or 'tv'")
):
    """Direct TMDB Item Details with official logos, cast, trailer, and stream pre-warming."""
    media_type = "tv" if type.lower() == "tv" else "movie"
    cache_key = f"tmdb_details_direct:{media_type}:{tmdb_id}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    client = getattr(app.state, "client", None)
    should_close = False
    if client is None:
        client = httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True)
        should_close = True

    try:
        # Parallel fetch TMDB details, credits, videos, and images
        t_main = client.get(f"https://api.themoviedb.org/3/{media_type}/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits,videos,images,recommendations", timeout=5.0)
        r_main = await t_main
        if r_main.status_code != 200:
            raise HTTPException(status_code=404, detail="TMDB content not found")

        data = r_main.json()
        title = data.get("title") or data.get("name") or "Unknown"
        year = data.get("release_date") or data.get("first_air_date") or ""
        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")
        vote_avg = data.get("vote_average", 0.0)

        # Smart English-First Logo Extraction
        raw_logos = data.get("images", {}).get("logos", [])
        all_logos = []
        for l in raw_logos:
            fp = l.get("file_path")
            if fp:
                all_logos.append({
                    "url": f"https://image.tmdb.org/t/p/original{fp}",
                    "url_w500": f"https://image.tmdb.org/t/p/w500{fp}",
                    "aspect_ratio": l.get("aspect_ratio"),
                    "width": l.get("width"),
                    "height": l.get("height"),
                    "lang": l.get("iso_639_1"),
                    "vote_average": l.get("vote_average"),
                    "vote_count": l.get("vote_count"),
                })

        NON_LATIN_LANGS = {"zh", "ja", "ko", "ar", "ru", "hi", "th", "he", "fa", "el", "ta", "te", "bn"}
        en_logos = [l for l in all_logos if l.get("lang") == "en"]
        picked_logo = None
        if en_logos:
            best_en = sorted(en_logos, key=lambda x: (x.get("vote_average", 0), x.get("vote_count", 0)), reverse=True)[0]
            top_global = all_logos[0]
            global_lang = top_global.get("lang")
            if global_lang not in NON_LATIN_LANGS and top_global.get("vote_average", 0) > best_en.get("vote_average", 0) + 1.0:
                picked_logo = top_global
            else:
                picked_logo = best_en
        elif all_logos:
            latin_logos = [l for l in all_logos if l.get("lang") not in NON_LATIN_LANGS]
            picked_logo = latin_logos[0] if latin_logos else all_logos[0]

        logo_url = picked_logo["url"] if picked_logo else None
        logo_w500 = picked_logo["url_w500"] if picked_logo else None

        # Extract YouTube Trailer
        trailer_url = None
        for v in data.get("videos", {}).get("results", []):
            if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                trailer_url = f"https://www.youtube.com/watch?v={v.get('key')}"
                break

        # Cast formatting
        cast_list = []
        for c in data.get("credits", {}).get("cast", [])[:10]:
            p_path = c.get("profile_path")
            cast_list.append({
                "name": c.get("name"),
                "character": c.get("character"),
                "profile_image": f"https://image.tmdb.org/t/p/w185{p_path}" if p_path else None
            })

        # Seasons shaping for TV
        seasons = []
        if media_type == "tv":
            for s in data.get("seasons", []):
                s_num = s.get("season_number")
                if s_num is not None and s_num > 0:
                    seasons.append({
                        "season_number": s_num,
                        "name": s.get("name"),
                        "episode_count": s.get("episode_count", 0),
                        "poster": f"https://image.tmdb.org/t/p/w500{s.get('poster_path')}" if s.get("poster_path") else None
                    })

        result = {
            "status": "success",
            "cached": False,
            "tmdb_id": tmdb_id,
            "media_type": media_type,
            "title": title,
            "original_title": data.get("original_title") or data.get("original_name"),
            "tagline": data.get("tagline", ""),
            "overview": data.get("overview", ""),
            "logo": logo_url,
            "logo_w500": logo_w500,
            "backdrop": f"https://image.tmdb.org/t/p/original{backdrop_path}" if backdrop_path else None,
            "poster": f"https://image.tmdb.org/t/p/original{poster_path}" if poster_path else None,
            "rating": round(vote_avg, 1) if vote_avg else None,
            "vote_count": data.get("vote_count", 0),
            "release_date": year,
            "runtime_minutes": data.get("runtime") or (data.get("episode_run_time", [None])[0] if data.get("episode_run_time") else None),
            "genres": [g.get("name") for g in data.get("genres", [])],
            "trailer": trailer_url,
            "cast": cast_list,
            "seasons": seasons if media_type == "tv" else None,
            "stream_url": f"/download/tmdb/{tmdb_id}?type={media_type}&season=1&episode=1"
        }
        _cache_set(cache_key, result, DETAILS_TTL)

        # Predictive background pre-warming of streams
        asyncio.create_task(_fetch_net27_stream_sources(tmdb_id=tmdb_id, is_series=(media_type == "tv"), se=1, ep=1))

        return result
    finally:
        if should_close:
            await client.aclose()


@app.get("/download/tmdb/{tmdb_id}")
async def get_tmdb_direct_stream(
    tmdb_id: int,
    type: str = Query("movie", description="Media type: 'movie' or 'tv'"),
    season: int = 1,
    episode: int = 1,
    se: int = 0,
    ep: int = 0,
):
    """Direct stream extraction via TMDB ID (1080p, 720p, 480p, 360p direct CDN MP4s)."""
    media_type = "tv" if type.lower() == "tv" else "movie"
    is_series = media_type == "tv"
    season_val = max(int(season or 1), int(se or 1)) if is_series else None
    episode_val = max(int(episode or 1), int(ep or 1)) if is_series else None

    cache_key = f"download_tmdb_direct:{media_type}:{tmdb_id}:{season_val}:{episode_val}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    # Parallel fetch streams and TMDB info
    net27_task = _fetch_net27_stream_sources(
        tmdb_id=tmdb_id,
        is_series=is_series,
        se=season_val or 1,
        ep=episode_val or 1
    )
    tmdb_task = get_tmdb_direct_details(tmdb_id=tmdb_id, type=media_type)

    net27_data, tmdb_info = await asyncio.gather(net27_task, tmdb_task, return_exceptions=True)
    if isinstance(tmdb_info, Exception):
        tmdb_info = {}

    files = []
    captions = []
    has_res = False

    if isinstance(net27_data, dict) and net27_data.get("streams"):
        has_res = True
        for s in net27_data.get("streams", []):
            url = s.get("url") or ""
            if not url:
                continue
            res_val = s.get("resolution")
            size_b = int(s.get("size", 0) or 0)
            files.append({
                "resolution": f"{res_val}p" if res_val else "unknown",
                "resolution_value": int(res_val) if str(res_val).isdigit() else 0,
                "size_bytes": size_b,
                "size_mb": round(size_b / (1024 * 1024), 2),
                "ext": "mp4",
                "id": str(s.get("id", "")),
                "stream_link": url,
                "codec": s.get("codec") or "h264",
                "vip_locked": False
            })
        files.sort(key=lambda x: x.get("resolution_value", 0), reverse=True)

        for c in net27_data.get("captions", []):
            c_url = c.get("url") or ""
            if "url=" in c_url:
                c_url = c_url.split("url=", 1)[-1]
                c_url = unquote(c_url)
            captions.append({
                "language": c.get("name") or c.get("lang"),
                "language_code": c.get("lang"),
                "url": c_url,
            })

    result = {
        "status": "success",
        "cached": False,
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "title": (tmdb_info or {}).get("title", "Unknown"),
        "logo": (tmdb_info or {}).get("logo"),
        "logo_w500": (tmdb_info or {}).get("logo_w500"),
        "backdrop": (tmdb_info or {}).get("backdrop"),
        "poster": (tmdb_info or {}).get("poster"),
        "season": season_val,
        "episode": episode_val,
        "has_resource": has_res,
        "qualities_count": len(files),
        "files": files,
        "subtitles": captions,
    }
    if has_res:
        _cache_set(cache_key, result, DETAILS_TTL)
        if is_series and episode_val:
            asyncio.create_task(_fetch_net27_stream_sources(tmdb_id=tmdb_id, is_series=True, se=season_val or 1, ep=episode_val + 1))

    return result


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=True)

