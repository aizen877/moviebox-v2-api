FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Render injects $PORT at runtime. Fall back to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Shell form so $PORT is expanded at container start.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT}
