# video-ranker

A multimodal video ranking system. Videos are ingested from public URLs, broken down into
keyframes, audio transcripts and metadata, embedded into a shared vector space, and ranked for a
query with a learned ranking model.

## Stack

- **Storage:** Postgres 17 + pgvector (metadata and embeddings), S3-compatible object storage
  (RustFS) for videos and keyframes, Redis for the task queue
- **Ingestion:** yt-dlp, ffmpeg
- **Models:** PyTorch (CUDA 12.8)

## Layout

```
src/video_ranker/
  config.py          settings loaded from .env
  storage.py         S3 helpers
  db/                SQLAlchemy models and session
  ingest/            download + registration pipeline
  cli.py             command-line entry point
tests/
docker-compose.yml   Postgres, S3, Redis
```

## Getting started

Requires Docker and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env              # then set real passwords
docker compose up -d --wait
uv sync                           # add --extra ml for PyTorch + CUDA
uv run video-ranker init-db
uv run video-ranker ingest "https://www.youtube.com/watch?v=..."
```

Run the checks:

```bash
uv run ruff check . && uv run pytest
```

## Roadmap

1. Ingestion and storage
2. Keyframe extraction, transcription, embeddings
3. Distributed processing with a task queue
4. Labelling and evaluation set
5. Retrieval and learned ranking
6. API and web UI
7. Monitoring and load testing
