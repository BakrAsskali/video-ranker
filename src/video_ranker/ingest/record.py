from video_ranker.db.models import Video


def build_record(video: Video, segments: list[dict] | None = None) -> dict:
    """One JSON-friendly dict per video: metadata, transcript and keyframes (no raw vectors)."""
    tr = video.transcript
    frames = sorted(video.keyframes, key=lambda f: f.timestamp_s)
    return {
        "id": video.id,
        "source_url": video.source_url,
        "platform": video.platform,
        "external_id": video.external_id,
        "title": video.title,
        "description": video.description,
        "duration_s": video.duration_s,
        "status": video.status,
        "video_key": video.video_key,
        "transcript": None
        if tr is None
        else {
            "language": tr.language,
            "text": tr.text,
            "embedded": tr.embedding is not None,
            "segments": segments or [],
        },
        "keyframes": [
            {
                "timestamp_s": f.timestamp_s,
                "image_key": f.image_key,
                "caption": f.caption,
                "embedded": f.embedding is not None,
            }
            for f in frames
        ],
    }
