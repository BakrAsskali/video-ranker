import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from video_ranker.ingest.caption import clean, pick_frames  # noqa: E402


def test_few_frames_all_picked():
    assert pick_frames(3, k=4) == [0, 1, 2]


def test_spread_keeps_ends():
    idx = pick_frames(20, k=4)
    assert len(idx) == 4
    assert idx[0] == 0 and idx[-1] == 19


def test_no_frames():
    assert pick_frames(0) == []


def test_clean_strips_boilerplate():
    assert clean(" In this image we can see a horse.") == "A horse."
