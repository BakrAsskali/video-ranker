from pathlib import Path

from sqlalchemy import select

from video_ranker.config import get_settings
from video_ranker.db.models import Video, VideoStatus
from video_ranker.db.session import SessionLocal
from video_ranker.ingest.download import download_video
from video_ranker.storage import ensure_bucket, upload_file


def ingest_url(url: str) -> int:
    """Download a video, store it in S3 and register it in Postgres. Returns the video id."""
    settings = get_settings()
    with SessionLocal() as session:
        video = session.scalar(select(Video).where(Video.source_url == url))
        if video is None:
            video = Video(source_url=url)
            session.add(video)
            session.commit()

        try:
            result = download_video(url, Path(settings.data_dir) / "raw")
            ensure_bucket()
            key = f"raw/{result.path.name}"
            upload_file(result.path, key)
            # Keep only the S3 copy locally to save disk; frames/audio come later.
            result.path.unlink(missing_ok=True)

            video.platform = result.platform
            video.external_id = result.external_id
            video.title = result.title
            video.description = result.description
            video.duration_s = result.duration_s
            video.video_key = key
            video.status = VideoStatus.DOWNLOADED
            video.error = None
        except Exception as exc:
            video.status = VideoStatus.FAILED
            video.error = str(exc)[:2000]
            session.commit()
            raise

        session.commit()
        return video.id
