"""
Moviebox Unofficial v2 API  -  FastAPI gateway for HuggingFace Spaces & Render.

Wraps the MovieBox H5 REST backend (h5-api.aoneroom.com) and exposes clean JSON endpoints.
Features automatic guest Bearer token acquisition, HTTP/2 multiplexing, ORJSON response engine,
search suggestions, catalog filtering, metadata details, stream extraction, and subtitle links.
"""

import asyncio
import contextvars
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from enum import Enum
from urllib.parse import quote

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

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

# Shared client limits & connection pool
_LIMITS = httpx.Limits(max_keepalive_connections=30, max_connections=60, keepalive_expiry=30.0)
_TIMEOUT = httpx.Timeout(20.0, connect=8.0)

# Global bearer token cache & lock
_bearer_token: str | None = None
_token_lock = asyncio.Lock()

# Ultra-fast in-memory TTL cache: key -> (expires_at, value)
_CACHE: dict[str, tuple[float, object]] = {}
HOMEPAGE_TTL = 300.0   # 5 min
SEARCH_TTL = 120.0     # 2 min
DETAILS_TTL = 600.0    # 10 min
RECOMMEND_TTL = 600.0  # 10 min


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
    # Initialize shared HTTP/2 AsyncClient with high-speed connection pool
    app.state.client = httpx.AsyncClient(
        http2=True,
        headers=DEFAULT_HEADERS,
        limits=_LIMITS,
        timeout=_TIMEOUT,
        follow_redirects=True
    )
    logger.info("Shared HTTP/2 httpx connection pool initialized.")
    # Pre-fetch bearer token on startup
    await _get_bearer_token()
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="MovieBox API Pro",
    description="Full Pure REST API for moviebox.ph — Ultra High Performance",
    version="2.2.0",
    lifespan=lifespan,
)

# ContextVar to store client IP per request context
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
    
    # Add Edge Cache headers for successful GET data requests
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
        return "103.191.240.1"  # Real public BD IP (AmberIT)
    elif host == "moviebox.ph":
        return "112.198.115.36"  # Real public PH IP (Globe Telecom)

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
                "/download/{slug_or_id}",
                "/api/stream/{subject_id}?detail_path=",
                "/api/stream/{subject_id}/captions?detail_path=",
                "/recommend/{slug_or_id}"
            ],
            "message": "High-Performance Pure REST API for moviebox.ph 🚀"
        })

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MovieBox API Pro | High-Performance REST Gateway</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #ff3d71;
                --secondary: #3366ff;
                --accent: #00f2ff;
                --bg: #07080c;
                --card-bg: rgba(255, 255, 255, 0.03);
                --glass: rgba(255, 255, 255, 0.06);
                --text: #ffffff;
            }

            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Outfit', sans-serif;
                background: var(--bg);
                color: var(--text);
                overflow-x: hidden;
                min-height: 100vh;
                background-image: 
                    radial-gradient(circle at 10% 10%, rgba(255, 61, 113, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 90% 90%, rgba(51, 102, 255, 0.12) 0%, transparent 40%);
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 60px 24px;
                position: relative;
            }

            header {
                text-align: center;
                margin-bottom: 80px;
                animation: fadeInDown 1s ease-out;
            }

            @keyframes fadeInDown {
                from { opacity: 0; transform: translateY(-30px); }
                to { opacity: 1; transform: translateY(0); }
            }

            h1 {
                font-size: clamp(2.5rem, 8vw, 4rem);
                font-weight: 800;
                background: linear-gradient(135deg, #fff 0%, #aaa 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 15px;
                letter-spacing: -2px;
            }

            .badge {
                background: linear-gradient(90deg, var(--primary), var(--secondary));
                padding: 8px 18px;
                border-radius: 40px;
                font-size: 0.85rem;
                font-weight: 700;
                display: inline-block;
                margin-bottom: 25px;
                text-transform: uppercase;
                letter-spacing: 1px;
                box-shadow: 0 10px 30px rgba(255, 61, 113, 0.3);
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
                gap: 30px;
                margin-top: 20px;
            }

            .card {
                background: var(--card-bg);
                border: 1px solid var(--glass);
                border-radius: 28px;
                padding: 35px;
                transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                backdrop-filter: blur(12px);
                position: relative;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }

            @media (hover: hover) {
                .card:hover {
                    transform: translateY(-12px) scale(1.02);
                    border-color: rgba(255,255,255,0.2);
                    box-shadow: 0 30px 60px rgba(0,0,0,0.5);
                }
            }

            .card-title {
                font-size: 1.5rem;
                font-weight: 700;
                margin-bottom: 18px;
                display: flex;
                align-items: center;
                gap: 12px;
            }

            .card-title i {
                width: 32px; height: 32px;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
                display: flex; align-items: center; justify-content: center;
                font-size: 1rem; color: var(--accent);
                font-style: normal;
            }

            .card-desc {
                color: #9ea3ac;
                font-size: 1rem;
                line-height: 1.6;
                margin-bottom: 25px;
                flex-grow: 1;
            }

            .endpoint {
                font-family: 'JetBrains Mono', monospace;
                background: rgba(0,0,0,0.4);
                padding: 14px;
                border-radius: 14px;
                font-size: 0.85rem;
                color: var(--accent);
                border: 1px solid rgba(0,242,255,0.15);
                margin-bottom: 25px;
                word-break: break-all;
                position: relative;
            }

            .endpoint::after {
                content: 'GET';
                position: absolute;
                right: 14px; top: 14px;
                font-size: 0.65rem; font-weight: 800;
                color: rgba(255,255,255,0.3);
            }

            .btn {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 16px;
                background: #ffffff;
                color: #000000;
                text-decoration: none;
                border-radius: 16px;
                font-weight: 700;
                font-size: 0.95rem;
                transition: all 0.3s;
            }

            .btn:hover {
                background: var(--primary);
                color: #fff;
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(255, 61, 113, 0.4);
            }

            footer {
                text-align: center;
                padding: 80px 0 40px;
                animation: fadeIn 2s ease;
            }

            @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

            .dev-tag {
                font-weight: 800;
                color: #666;
                letter-spacing: 3px;
                text-transform: uppercase;
                font-size: 0.75rem;
                border: 1px solid #222;
                padding: 12px 30px;
                border-radius: 50px;
                display: inline-block;
                background: rgba(255,255,255,0.01);
                transition: all 0.3s;
            }

            .dev-tag:hover {
                color: var(--text);
                border-color: var(--primary);
                letter-spacing: 5px;
            }

            @media (max-width: 480px) {
                .container { padding: 40px 16px; }
                .card { padding: 25px; }
                h1 { margin-bottom: 10px; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="badge">Ultra-Fast Edition &bull; HTTP/2 &bull; Rust JSON</div>
                <h1>MovieBox Pro</h1>
                <p style="color: #889; font-size: 1.25rem; font-weight: 300;">State-of-the-Art Pure REST Architecture</p>
                <div style="margin-top: 15px;">
                    <a href="/docs" target="_blank" style="color: var(--accent); text-decoration: none; font-size: 0.95rem; font-weight: 600; margin-right: 20px;">Swagger Docs &rarr;</a>
                    <a href="/redoc" target="_blank" style="color: var(--primary); text-decoration: none; font-size: 0.95rem; font-weight: 600;">ReDoc &rarr;</a>
                </div>
            </header>

            <div class="grid">
                <div class="card">
                    <div class="card-title"><i>🏠</i> Discover Home</div>
                    <p class="card-desc">The ultimate window into MovieBox. Headlines, recommended content, and trending blocks updated in real-time.</p>
                    <div class="endpoint">/home</div>
                    <a href="/home" target="_blank" class="btn">Launch API</a>
                </div>

                <div class="card">
                    <div class="card-title"><i>🔍</i> Smart Search</div>
                    <p class="card-desc">High-precision search engine results. Returns titles, posters, and slugs for lightning-fast matching.</p>
                    <div class="endpoint">/search?q=Attack on Titan</div>
                    <a href="/search?q=Attack on Titan" target="_blank" class="btn">Test Search</a>
                </div>

                <div class="card">
                    <div class="card-title"><i>💡</i> Search Autocomplete</div>
                    <p class="card-desc">Instant keyword suggestions as you type for search boxes and live autocomplete fields.</p>
                    <div class="endpoint">/search/suggest?q=batman</div>
                    <a href="/search/suggest?q=batman" target="_blank" class="btn">Test Suggestions</a>
                </div>

                <div class="card">
                    <div class="card-title"><i>🆔</i> Metadata A-Z</div>
                    <p class="card-desc">Deep-dive into any subject. Episodes, seasons, languages, and full high-resolution metadata trees.</p>
                    <div class="endpoint">/details/{slug_or_id}</div>
                    <a href="/details/attack-on-titan-hindi-kGWQOIx0d4" target="_blank" class="btn">Fetch Specs</a>
                </div>

                <div class="card">
                    <div class="card-title"><i>🎬</i> Unified Download & Streams</div>
                    <p class="card-desc">All stream qualities + subtitles resolved in a single call. Works with both movies and multi-season TV shows.</p>
                    <div class="endpoint">/download/{slug}?se=1&ep=1</div>
                    <a href="/download/attack-on-titan-hindi-kGWQOIx0d4?se=1&ep=1" target="_blank" class="btn">Get Streams & Subs</a>
                </div>

                <div class="card">
                    <div class="card-title"><i>📦</i> Catalog Filters</div>
                    <p class="card-desc">Paginated collections for all genres. Movies, TV shows, and Animations filtered by professional criteria.</p>
                    <div class="endpoint">/tv-series?page=1</div>
                    <a href="/tv-series?page=1" target="_blank" class="btn">Browse TV Series</a>
                </div>
            </div>

            <footer>
                <div class="dev-tag">MovieBox Unofficial v2 API</div>
            </footer>
        </div>
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


async def _fetch_details(detail_path: str) -> dict:
    """Fetch + cache raw item details via the shared client."""
    cache_key = f"details:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    is_numeric = str(detail_path).isdigit()
    params = {"subjectId": detail_path} if is_numeric else {"detailPath": detail_path}
    data = await _api_get("/wefeed-h5api-bff/detail", params)
    _cache_set(cache_key, data, DETAILS_TTL)
    return data


@app.get("/details/{detail_path}")
@app.get("/detail/{detail_path}")
async def get_details(detail_path: str):
    """Specific item details (id = subjectId or detailPath, cached 10 min)."""
    cache_key = f"details_resp:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}
    try:
        data = await _fetch_details(detail_path)
        result = {
            "status": "success",
            "cached": False,
            "detail_path": detail_path,
            "seasons": _shape_seasons(data),
            "dubs": _shape_dubs(data),
            "data": data,
        }
        _cache_set(cache_key, result, DETAILS_TTL)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching details for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


def _shape_download(detail_path, subject, dl_data):
    downloads = (dl_data or {}).get("downloads", []) or []
    captions = (dl_data or {}).get("captions", []) or []

    def _ext(url: str) -> str:
        path = (url or "").split("?")[0]
        return path.rsplit(".", 1)[-1] if "." in path else ""

    files = [
        {
            "resolution": f"{m.get('resolution')}p",
            "resolution_value": m.get("resolution"),
            "size_bytes": int(m.get("size", 0) or 0),
            "size_mb": round(int(m.get("size", 0) or 0) / (1024 * 1024), 2),
            "ext": _ext(m.get("url", "")),
            "id": m.get("id", ""),
            "stream_link": m.get("url", ""),
        }
        for m in downloads
    ]
    subtitles = [
        {
            "language": c.get("lanName") or c.get("lan"),
            "language_code": c.get("lan"),
            "size_bytes": int(c.get("size", 0) or 0),
            "delay": c.get("delay", 0),
            "url": c.get("url"),
        }
        for c in captions
    ]

    subj = subject or {}
    cover = subj.get("cover") or {}
    subject_type = subj.get("subjectType")
    return {
        "status": "success",
        "detail_path": detail_path,
        "subject_id": subj.get("subjectId"),
        "title": subj.get("title", "Unknown"),
        "subject_type": _SUBJECT_TYPE_NAME.get(subject_type, str(subject_type)),
        "release_date": subj.get("releaseDate", ""),
        "cover_image": cover.get("url"),
        "has_resource": (dl_data or {}).get("hasResource", False),
        "limited": (dl_data or {}).get("limited", False),
        "qualities_count": len(files),
        "files": files,
        "subtitles": subtitles,
    }


@app.get("/download/{detail_path}")
async def get_download_links(
    detail_path: str,
    season: int = Query(0, ge=0, description="Season number (TV series only)"),
    episode: int = Query(0, ge=0, description="Episode number (TV series only)"),
):
    """All available stream / download links + subtitles in a single request."""
    cache_key = f"download:{detail_path}:{season}:{episode}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        details_data = await _fetch_details(detail_path)
        subject = (details_data or {}).get("subject") or {}
        subject_id = subject.get("subjectId")
        detail_path_slug = subject.get("detailPath")

        if not subject_id or not detail_path_slug:
            raise HTTPException(status_code=404, detail="Could not resolve subject details")

        se_val = season if isinstance(season, int) else 0
        ep_val = episode if isinstance(episode, int) else 0

        is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
        se = se_val
        ep = ep_val
        if is_series and se_val == 0 and ep_val == 0:
            se = 1
            ep = 1

        params = {
            "subjectId": str(subject_id),
            "detailPath": str(detail_path_slug),
            "se": se,
            "ep": ep,
        }

        dl_data = await _api_get("/wefeed-h5api-bff/subject/download", params)

        result = _shape_download(detail_path, subject, dl_data)
        result["cached"] = False
        _cache_set(cache_key, result, DETAILS_TTL)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching download links for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stream/{subject_id}")
async def get_stream_sources(subject_id: str, detail_path: str = "", se: int = 1, ep: int = 1):
    """Direct stream engine integration via media player domain."""
    try:
        dom_data = await _api_get("/wefeed-h5api-bff/media-player/get-domain")
        domain = str(dom_data if isinstance(dom_data, str) else dom_data.get("data", "https://netfilm.world")).rstrip("/")
        if not domain.startswith("http"):
            domain = "https://netfilm.world"

        player_referer = (
            f"{domain}/spa/videoPlayPage/movies/{detail_path}"
            f"?id={subject_id}&type=/movie/detail&detailSe={se}&detailEp={ep}&lang=en"
        )
        play_url = f"{domain}/wefeed-h5api-bff/subject/play?subjectId={subject_id}&se={se}&ep={ep}&detailPath={detail_path}"

        token = await _get_bearer_token()
        headers = {
            **PLAYER_HEADERS,
            "Referer": player_referer,
            "Authorization": f"Bearer {token}" if token else ""
        }

        client = getattr(app.state, "client", None)
        if client is None:
            async with httpx.AsyncClient(http2=True, timeout=_TIMEOUT, follow_redirects=True) as temp_client:
                resp = await temp_client.get(play_url, headers=headers)
        else:
            resp = await client.get(play_url, headers=headers)

        data = resp.json().get("data", {}) if resp.status_code == 200 else {}

        has_resource = data.get("hasResource", False)
        streams = [
            {
                "resolution": f"{s.get('resolutions')}p",
                "format": s.get("format"),
                "url": s.get("url"),
                "size": s.get("size"),
                "duration": s.get("duration"),
                "codec": s.get("codecName")
            }
            for s in data.get("streams", [])
        ]
        return {
            "status": "success",
            "subject_id": subject_id,
            "detail_path": detail_path,
            "se": se,
            "ep": ep,
            "has_resource": has_resource,
            "sources": streams,
            "hls": data.get("hls", []),
            "dash": data.get("dash", []),
            "free_episodes": data.get("freeNum"),
            "limited": data.get("limited", False),
            "note": None if has_resource else "No stream found for this episode."
        }
    except Exception as e:
        logger.error(f"Error in stream extraction: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stream/{subject_id}/captions")
async def get_captions(subject_id: str, detail_path: str = "", se: int = 1, ep: int = 1):
    """Fetch captions for a stream subject."""
    try:
        dom_data = await _api_get("/wefeed-h5api-bff/media-player/get-domain")
        domain = str(dom_data if isinstance(dom_data, str) else dom_data.get("data", "https://netfilm.world")).rstrip("/")
        if not domain.startswith("http"):
            domain = "https://netfilm.world"

        player_referer = (
            f"{domain}/spa/videoPlayPage/movies/{detail_path}"
            f"?id={subject_id}&type=/movie/detail&detailSe={se}&detailEp={ep}&lang=en"
        )
        play_url = f"{domain}/wefeed-h5api-bff/subject/play?subjectId={subject_id}&se={se}&ep={ep}&detailPath={detail_path}"

        token = await _get_bearer_token()
        headers = {
            **PLAYER_HEADERS,
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

        stream_id = None
        stream_format = None
        if streams:
            stream_id = streams[0].get("id")
            stream_format = streams[0].get("format", "MP4")
        elif dash:
            stream_id = dash[0].get("id")
            stream_format = dash[0].get("format", "DASH")

        if not stream_id:
            return {"status": "success", "subject_id": subject_id, "se": se, "ep": ep, "count": 0, "captions": []}

        cap_url = (
            f"/wefeed-h5api-bff/subject/caption"
            f"?format={stream_format}&id={stream_id}&subjectId={subject_id}&detailPath={detail_path}"
        )
        data = await _api_get(cap_url)
        captions = data.get("captions", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        return {"status": "success", "subject_id": subject_id, "se": se, "ep": ep, "count": len(captions), "captions": captions}
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


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=True)
