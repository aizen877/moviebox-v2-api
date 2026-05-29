"""
Moviebox Unofficial v2 API  -  FastAPI gateway for HuggingFace Spaces.

Wraps the moviebox_api.v2 client (H5 REST backend, h5-api.aoneroom.com) and
exposes clean JSON endpoints. Returns EVERY stream quality + all subtitles in a
single /download request.

Performance:
    * ONE shared httpx.AsyncClient (keep-alive connection pool) for the hot
      read paths instead of a fresh client per request (~6x faster).
    * gzip transfer-encoding (583 KB -> ~90 KB on the wire).
    * Small in-memory TTL cache for homepage/search/details (instant repeats).

Endpoints:
    GET /                      -> service info
    GET /health                -> liveness probe
    GET /homepage              -> landing page content (cached)
    GET /search?q=&type=&page= -> search movies / tv / etc (cached)
    GET /details/{id}          -> item details (id = subjectId or detailPath)
    GET /download/{id}?se=&ep= -> all stream links + subtitles
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query

from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from moviebox_api.v1.constants import SubjectType
from moviebox_api.v2.constants import HOST_URL, REFERER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("moviebox_v2_api")

# --- fast HTTP layer ---------------------------------------------------------

BASE = HOST_URL.rstrip("/")
# "More like this" / recommendations live on the v1 host under a different
# path prefix (/wefeed-h5-bff/web). The v2 host (h5-api) 404s on it.
REC_BASE = "https://h5.aoneroom.com"
FAST_HEADERS = {
    "X-Client-Info": '{"timezone":"Asia/Dhaka"}',
    "Accept-Language": "bn-BD,bn;q=0.9,en-US;q=0.8,en;q=0.5",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0",
    "Referer": REFERER,
    "Origin": REFERER,
}

# Shared client + limits => connection pooling / keep-alive across requests.
_LIMITS = httpx.Limits(max_keepalive_connections=20, max_connections=50)
_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# Tiny TTL cache: key -> (expires_at, value)
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient(
        headers=FAST_HEADERS, limits=_LIMITS, timeout=_TIMEOUT, follow_redirects=True
    )
    logger.info("Shared httpx client ready.")
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(
    title="Moviebox Unofficial v2 API",
    description="High-speed Python gateway for the MovieBox H5 REST backend. "
    "All stream qualities + subtitles in a single request.",
    version="2.1.0",
    lifespan=lifespan,
)

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


# subjectType int -> readable name (for download response)
_SUBJECT_TYPE_NAME = {
    SubjectType.ALL.value: "ALL",
    SubjectType.MOVIES.value: "MOVIES",
    SubjectType.TV_SERIES.value: "TV_SERIES",
    SubjectType.EDUCATION.value: "EDUCATION",
    SubjectType.MUSIC.value: "MUSIC",
    SubjectType.ANIME.value: "ANIME",
}


async def _api_get(path: str, params: dict | None = None) -> dict | list:
    """GET an H5 endpoint via the shared client and unwrap the data envelope."""
    r = await app.state.client.get(BASE + path, params=params or {})
    r.raise_for_status()
    j = r.json()
    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j["data"]
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


async def _rec_get(path: str, params: dict | None = None) -> dict | list:
    """GET an endpoint on the recommendation host (h5.aoneroom.com)."""
    r = await app.state.client.get(REC_BASE + path, params=params or {})
    r.raise_for_status()
    j = r.json()
    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j["data"]
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


async def _api_post(path: str, json_body: dict) -> dict | list:
    r = await app.state.client.post(BASE + path, json=json_body)
    r.raise_for_status()
    j = r.json()
    if j.get("code", 1) == 0 and j.get("message") == "ok":
        return j["data"]
    raise HTTPException(status_code=502, detail=f"Upstream error: {j.get('message')!r}")


# --- routes ------------------------------------------------------------------


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Moviebox Unofficial API (v2 H5 REST Backend)",
        "docs": "/docs",
        "endpoints": ["/homepage", "/search?q=", "/details/{id}", "/download/{id}", "/recommend/{id}"],
        "message": "All stream qualities + subtitles in a single request. 🚀",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/homepage")
async def get_homepage(
    host: str = Query(
        "moviebox.com.bd",
        description="MovieBox content host/region (e.g. moviebox.com.bd, moviebox.ph)",
    ),
):
    """Landing-page content listings (cached 5 min).

    Note: MovieBox decides content region primarily from the server egress IP,
    not this `host` param. It's exposed for flexibility but may not change much.
    """
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


@app.get("/search")
async def search(
    q: str = Query(..., description="Search keyword"),
    type: str = Query(
        "all",
        description="Content type (all, movies, tv_series, anime, music, education)",
    ),
    page: int = Query(1, ge=1, description="Page number"),
):
    """Search movies / tv-series / music etc (cached 2 min)."""
    subject_type = _map_subject_type(type)
    cache_key = f"search:{subject_type.value}:{page}:{q.lower().strip()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {"status": "success", "cached": True, **cached}

    try:
        data = await _api_post(
            "/wefeed-h5api-bff/subject/search",
            {
                "keyword": q,
                "page": page,
                "perPage": 24,
                "subjectType": subject_type.value,
            },
        )
        items = (data or {}).get("items", []) or []
        if subject_type is not SubjectType.ALL:
            items = [it for it in items if it.get("subjectType") == subject_type.value]
            data["items"] = items

        payload = {"query": q, "type": type, "page": page, "data": data}
        _cache_set(cache_key, payload, SEARCH_TTL)
        return {"status": "success", "cached": False, **payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching '{q}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


def _shape_seasons(details_data: dict) -> dict:
    """Build a clean season/episode summary from a raw /detail payload.

    Upstream stores this under `data.resource.seasons`, where each entry is::

        {"se": 1, "maxEp": 13, "allEp": "", "resolutions": [{"resolution": 360, "epNum": 13}, ...]}

    Movies have no seasons -> returns is_series=False with empty list.
    """
    data = details_data or {}
    subject = data.get("subject") or {}
    is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
    raw_seasons = ((data.get("resource") or {}).get("seasons")) or []

    seasons = []
    for s in raw_seasons:
        resolutions = s.get("resolutions") or []
        # episodes available = highest epNum across resolutions, fallback to maxEp
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


@app.get("/details/{detail_path}")
async def get_details(detail_path: str):
    """Specific item details (id = subjectId or detailPath, cached 10 min).

    For TV series, a clean `seasons` summary (season count + episodes per season
    + available resolutions) is added alongside the raw upstream `data`.
    """
    cache_key = f"details:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {
            "status": "success",
            "cached": True,
            "detail_path": detail_path,
            "seasons": _shape_seasons(cached),
            "data": cached,
        }
    try:
        data = await _fetch_details(detail_path)
        return {
            "status": "success",
            "cached": False,
            "detail_path": detail_path,
            "seasons": _shape_seasons(data),
            "data": data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching details for '{detail_path}': {e}")
        raise HTTPException(status_code=502, detail=str(e))


async def _fetch_details(detail_path: str) -> dict:
    """Fetch + cache raw item details via the shared client."""
    cache_key = f"details:{detail_path}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    is_numeric = detail_path.isdigit()
    params = {"subjectId": detail_path} if is_numeric else {"detailPath": detail_path}
    data = await _api_get("/wefeed-h5api-bff/detail", params)
    _cache_set(cache_key, data, DETAILS_TTL)
    return data


def _download_params(identifier: str, season: int, episode: int) -> dict:
    """Build /subject/download params, routing numeric ids to the subjectId slot.

    The upstream endpoint expects EITHER a non-empty `subjectId` OR a non-empty
    `detailPath` (passing a numeric subjectId in the detailPath slot returns a
    400 'empty subjectId' error). detailPaths look like 'titans-q33meCQkvT7'.
    """
    if identifier.isdigit():
        return {"subjectId": identifier, "se": season, "ep": episode, "detailPath": ""}
    return {"subjectId": "", "se": season, "ep": episode, "detailPath": identifier}


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
            "stream_link": m.get("url"),
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
    """All available stream / download links + subtitles in a single request.

    Fast path: the /detail and /download calls run in PARALLEL on the shared
    keep-alive connection pool. /download works directly from detailPath, so we
    don't need to wait for /detail to discover the subjectId. Result is cached.
    """
    cache_key = f"download:{detail_path}:{season}:{episode}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        # Kick off both requests concurrently.
        details_task = asyncio.create_task(_fetch_details(detail_path))
        dl_task = asyncio.create_task(
            _api_get(
                "/wefeed-h5api-bff/subject/download",
                _download_params(detail_path, season, episode),
            )
        )

        details_data = await details_task
        subject = (details_data or {}).get("subject") or {}

        # If it's a tv-series and the caller didn't pass se/ep, re-fetch E1.
        is_series = subject.get("subjectType") == SubjectType.TV_SERIES.value
        if is_series and season == 0 and episode == 0:
            dl_task.cancel()
            dl_data = await _api_get(
                "/wefeed-h5api-bff/subject/download",
                _download_params(detail_path, 1, 1),
            )
        else:
            dl_data = await dl_task

        result = _shape_download(detail_path, subject, dl_data)
        result["cached"] = False
        _cache_set(cache_key, result, DETAILS_TTL)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching download links for '{detail_path}': {e}")
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
    """"More like this" - related movies / series for a given item.

    `id` may be a numeric subjectId or a detailPath. A detailPath is resolved to
    its subjectId first (the upstream rec endpoint needs the numeric id). Cached.
    """
    cache_key = f"recommend:{detail_path}:{page}:{per_page}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return {**cached, "cached": True}

    try:
        subject_id = detail_path
        if not detail_path.isdigit():
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

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
