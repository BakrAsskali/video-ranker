from dataclasses import dataclass
from pathlib import Path

import yt_dlp


@dataclass
class DownloadResult:
    path: Path
    platform: str | None
    external_id: str | None
    title: str | None
    description: str | None
    duration_s: float | None


def download_video(url: str, out_dir: Path, max_height: int = 720) -> DownloadResult:
    """Download a single video with yt-dlp, capped at `max_height` to save disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "%(extractor)s_%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    return DownloadResult(
        path=path,
        platform=info.get("extractor_key"),
        external_id=info.get("id"),
        title=info.get("title"),
        description=info.get("description"),
        duration_s=info.get("duration"),
    )
