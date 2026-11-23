import ctypes
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from video_ranker.config import get_settings


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    language: str | None
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)


def has_audio(video: Path) -> bool:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(out.stdout.strip())


def extract_audio(video: Path, out: Path) -> Path:
    """Decode the audio track to 16 kHz mono WAV, the format Whisper expects."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(out),
        ],
        check=True,
    )
    return out


def _preload_cuda_libs() -> None:
    """Load cuBLAS/cuDNN from the pip-installed NVIDIA wheels so CTranslate2 can find them.

    The wheels pulled in by torch aren't on the dynamic loader path, and CTranslate2 looks
    them up by soname. Loading them globally first makes those lookups succeed.
    """
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return
    for pkg, names in (
        (nvidia.cublas.lib, ("libcublasLt.so.12", "libcublas.so.12")),
        (nvidia.cudnn.lib, ("libcudnn.so.9",)),
    ):
        for name in names:
            path = os.path.join(next(iter(pkg.__path__)), name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


@lru_cache
def load_model():
    from faster_whisper import WhisperModel

    s = get_settings()
    if s.whisper_device != "cpu":
        _preload_cuda_libs()
    if s.hf_token:
        # huggingface_hub reads the token from the environment when fetching weights.
        os.environ.setdefault("HF_TOKEN", s.hf_token)
    return WhisperModel(s.whisper_model, device=s.whisper_device, compute_type=s.whisper_compute)


def transcribe(audio: Path) -> TranscriptResult:
    segments, info = load_model().transcribe(str(audio), vad_filter=True, beam_size=5)
    return TranscriptResult(
        language=info.language,
        segments=[Segment(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segments],
    )
