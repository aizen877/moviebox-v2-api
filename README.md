# Moviebox Unofficial v2 API

High-speed FastAPI gateway over the MovieBox H5 REST backend
(`h5-api.aoneroom.com`). Returns **every stream quality + all subtitles in a
single request**.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness probe |
| GET | `/docs` | Interactive Swagger UI |
| GET | `/homepage` | Landing-page content |
| GET | `/search?q=Titanic&type=movies&page=1` | Search |
| GET | `/details/{id}` | Item details (`id` = subjectId or detailPath) + seasons summary |
| GET | `/download/{id}?season=1&episode=1` | All stream links + subtitles |

### Examples

```
/search?q=Titanic&type=movies
/details/merlin-sMxCiIO6fZ9
/download/merlin-sMxCiIO6fZ9?season=1&episode=2     # tv-series
```

`type` accepts: `all, movies, tv_series, anime, music, education`.

## Deploy on Render

This repo is Render-ready (Docker). Render injects `$PORT` at runtime and the
`Dockerfile` binds uvicorn to it. Either:

- Click **New + → Web Service**, connect this repo, choose **Docker**, plan **Free**, or
- Use the included `render.yaml` (Blueprint) for one-click setup.

Health check path: `/health`.

> Note: Free Render services spin down after ~15 min of inactivity and take
> about a minute to wake on the next request.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
