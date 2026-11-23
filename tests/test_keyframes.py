from video_ranker.ingest.keyframes import pick_timestamps


def test_uniform_fallback_without_scene_cuts():
    assert pick_timestamps([], duration=10.0, min_frames=4) == [2.0, 4.0, 6.0, 8.0]


def test_close_timestamps_are_merged():
    ts = pick_timestamps([1.0, 1.2, 1.3], duration=10.0, min_frames=1)
    assert ts == [1.0, 5.0]


def test_frame_count_is_capped():
    scenes = [i * 0.6 for i in range(100)]
    assert len(pick_timestamps(scenes, duration=60.0, max_frames=32)) == 32
