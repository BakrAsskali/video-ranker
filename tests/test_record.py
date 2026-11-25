import json

from video_ranker.db.models import Keyframe, Transcript, Video
from video_ranker.ingest.record import build_record


def make_video():
    v = Video(id=1, source_url="https://example.com/v", title="zoo", status="processed")
    v.keyframes = [
        Keyframe(timestamp_s=5.0, image_key="keyframes/1/b.jpg", caption="an elephant"),
        Keyframe(timestamp_s=0.0, image_key="keyframes/1/a.jpg", embedding=[0.0] * 768),
    ]
    v.transcript = Transcript(language="en", text="hello zoo")
    return v


def test_record_shape():
    rec = build_record(make_video(), [{"start": 0.0, "end": 1.0, "text": "hello zoo"}])
    assert rec["title"] == "zoo"
    assert [f["timestamp_s"] for f in rec["keyframes"]] == [0.0, 5.0]
    assert rec["keyframes"][0]["embedded"] and not rec["keyframes"][1]["embedded"]
    assert rec["transcript"]["segments"][0]["text"] == "hello zoo"
    json.dumps(rec)


def test_record_without_transcript():
    v = Video(id=2, source_url="https://example.com/w")
    assert build_record(v)["transcript"] is None
    assert build_record(v)["keyframes"] == []
