import re
import subprocess
from pathlib import Path

_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def probe_duration(video: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def scene_timestamps(video: Path, threshold: float = 0.3) -> list[float]:
    """Timestamps (s) where ffmpeg's scene score exceeds `threshold`."""
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-vf",
            f"select='gt(scene,{threshold})',showinfo",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [float(m) for m in _PTS_RE.findall(proc.stderr)]


def pick_timestamps(
    scenes: list[float], duration: float, min_frames: int = 4, max_frames: int = 32
) -> list[float]:
    """Merge scene cuts with evenly spaced samples, de-duplicate and cap the count."""
    step = duration / (min_frames + 1)
    uniform = [step * (i + 1) for i in range(min_frames)]
    merged = sorted(set(round(t, 2) for t in scenes + uniform if 0 <= t < duration))

    deduped: list[float] = []
    for t in merged:
        if not deduped or t - deduped[-1] >= 0.5:
            deduped.append(t)

    if len(deduped) > max_frames:
        stride = len(deduped) / max_frames
        deduped = [deduped[int(i * stride)] for i in range(max_frames)]
    return deduped


def extract_frame(video: Path, timestamp: float, out: Path, width: int = 384) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.2f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "3",
            str(out),
        ],
        check=True,
    )
    return out


def extract_keyframes(video: Path, out_dir: Path) -> list[tuple[float, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video)
    timestamps = pick_timestamps(scene_timestamps(video), duration)
    return [(t, extract_frame(video, t, out_dir / f"{int(t * 1000):08d}.jpg")) for t in timestamps]
