import json
import tempfile
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import select

from video_ranker.config import get_settings
from video_ranker.db.models import Keyframe, Transcript, Video, VideoStatus
from video_ranker.db.session import SessionLocal
from video_ranker.ingest.download import download_video
from video_ranker.ingest.keyframes import extract_keyframes
from video_ranker.ingest.record import build_record
from video_ranker.ingest.transcribe import TranscriptResult, extract_audio, has_audio, transcribe
from video_ranker.storage import (
    download_bytes,
    download_file,
    ensure_bucket,
    upload_bytes,
    upload_file,
)


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


def extract_video_keyframes(video_id: int) -> int:
    """Pull a video from S3, extract keyframes, upload them and store the rows."""
    with SessionLocal() as session, tempfile.TemporaryDirectory() as tmp:
        video = session.get(Video, video_id)
        if video is None or video.video_key is None:
            raise ValueError(f"video {video_id} has no stored file")

        local = download_file(video.video_key, Path(tmp) / "video.mp4")
        frames = extract_keyframes(local, Path(tmp) / "frames")

        video.keyframes.clear()
        for timestamp, path in frames:
            key = upload_file(path, f"keyframes/{video.id}/{path.name}")
            video.keyframes.append(Keyframe(timestamp_s=timestamp, image_key=key))
        session.commit()
        return len(frames)


def keyframes_for_pending() -> Iterator[tuple[int, int]]:
    """Extract keyframes for every downloaded video that doesn't have any yet."""
    with SessionLocal() as session:
        ids = session.scalars(
            select(Video.id).where(Video.status == VideoStatus.DOWNLOADED, ~Video.keyframes.any())
        ).all()
    for video_id in ids:
        yield video_id, extract_video_keyframes(video_id)


def transcribe_video(video_id: int) -> int:
    """Transcribe a stored video. Full text goes to Postgres, timed segments to S3 as JSON."""
    with SessionLocal() as session, tempfile.TemporaryDirectory() as tmp:
        video = session.get(Video, video_id)
        if video is None or video.video_key is None:
            raise ValueError(f"video {video_id} has no stored file")

        local = download_file(video.video_key, Path(tmp) / "video.mp4")
        if has_audio(local):
            result = transcribe(extract_audio(local, Path(tmp) / "audio.wav"))
        else:
            result = TranscriptResult(language=None)

        payload = [vars(s) for s in result.segments]
        upload_bytes(json.dumps(payload).encode(), f"transcripts/{video.id}.json")

        video.transcript = Transcript(language=result.language, text=result.text)
        session.commit()
        return len(result.segments)


def transcribe_pending() -> Iterator[tuple[int, int]]:
    """Transcribe every downloaded video that has no transcript yet."""
    with SessionLocal() as session:
        ids = session.scalars(
            select(Video.id).where(Video.status == VideoStatus.DOWNLOADED, ~Video.transcript.has())
        ).all()
    for video_id in ids:
        yield video_id, transcribe_video(video_id)


def embed_video(video_id: int) -> int:
    """Fill in SigLIP embeddings for a video's keyframes and transcript."""
    from video_ranker.ingest.embed import embed_document, embed_images

    with SessionLocal() as session, tempfile.TemporaryDirectory() as tmp:
        video = session.get(Video, video_id)
        if video is None:
            raise ValueError(f"video {video_id} not found")

        frames = [k for k in video.keyframes if k.embedding is None]
        if frames:
            paths = [download_file(k.image_key, Path(tmp) / f"{k.id}.jpg") for k in frames]
            for frame, vec in zip(frames, embed_images(paths), strict=True):
                frame.embedding = vec.tolist()

        tr = video.transcript
        if tr is not None and tr.embedding is None:
            vec = embed_document(tr.text)
            if vec is not None:
                tr.embedding = vec.tolist()

        session.commit()
        return len(frames)


def embed_pending() -> Iterator[tuple[int, int]]:
    with SessionLocal() as session:
        ids = session.scalars(
            select(Video.id).where(
                Video.keyframes.any(Keyframe.embedding.is_(None))
                | Video.transcript.has(Transcript.embedding.is_(None))
            )
        ).all()
    for video_id in ids:
        yield video_id, embed_video(video_id)


def caption_video(video_id: int, k: int = 4) -> int:
    """Caption a handful of evenly spread keyframes with the VLM."""
    from video_ranker.ingest.caption import caption_images, pick_frames

    with SessionLocal() as session, tempfile.TemporaryDirectory() as tmp:
        video = session.get(Video, video_id)
        if video is None:
            raise ValueError(f"video {video_id} not found")

        frames = sorted(video.keyframes, key=lambda f: f.timestamp_s)
        chosen = [frames[i] for i in pick_frames(len(frames), k)]
        paths = [download_file(f.image_key, Path(tmp) / f"{f.id}.jpg") for f in chosen]
        for frame, text in zip(chosen, caption_images(paths), strict=True):
            frame.caption = text
        session.commit()
        return len(chosen)


def caption_pending() -> Iterator[tuple[int, int]]:
    with SessionLocal() as session:
        ids = session.scalars(
            select(Video.id).where(
                Video.keyframes.any(), ~Video.keyframes.any(Keyframe.caption.is_not(None))
            )
        ).all()
    for video_id in ids:
        yield video_id, caption_video(video_id)


def write_record(video_id: int) -> dict:
    with SessionLocal() as session:
        video = session.get(Video, video_id)
        segments = None
        if video.transcript is not None:
            segments = json.loads(download_bytes(f"transcripts/{video.id}.json"))
        rec = build_record(video, segments)
    upload_bytes(json.dumps(rec, indent=2).encode(), f"records/{video_id}.json")
    return rec


def process_video(video_id: int) -> dict:
    """Run every missing step for one video, mark it processed and write its record."""
    with SessionLocal() as session:
        video = session.get(Video, video_id)
        has_frames = bool(video.keyframes)
        has_transcript = video.transcript is not None
        has_captions = any(f.caption for f in video.keyframes)

    try:
        if not has_frames:
            extract_video_keyframes(video_id)
        if not has_transcript:
            transcribe_video(video_id)
        embed_video(video_id)
        if not has_captions:
            caption_video(video_id)
    except Exception as exc:
        with SessionLocal() as session:
            video = session.get(Video, video_id)
            video.status = VideoStatus.FAILED
            video.error = str(exc)[:2000]
            session.commit()
        raise

    with SessionLocal() as session:
        video = session.get(Video, video_id)
        video.status = VideoStatus.PROCESSED
        video.error = None
        session.commit()
    return write_record(video_id)


def process_pending() -> Iterator[int]:
    with SessionLocal() as session:
        ids = session.scalars(select(Video.id).where(Video.status == VideoStatus.DOWNLOADED)).all()
    for video_id in ids:
        process_video(video_id)
        yield video_id
