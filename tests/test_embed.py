import pytest

pytest.importorskip("torch")
pytest.importorskip("open_clip")

from video_ranker.ingest.embed import chunk_words  # noqa: E402


def test_short_text_is_one_chunk():
    assert chunk_words("hello there") == ["hello there"]


def test_empty_text():
    assert chunk_words("   ") == []


def test_long_text_overlaps():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_words(" ".join(words), size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "w99"
