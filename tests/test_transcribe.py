from video_ranker.ingest.transcribe import Segment, TranscriptResult


def test_text_joins_non_empty_segments():
    result = TranscriptResult(
        language="en",
        segments=[Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, ""), Segment(2.0, 3.0, "world")],
    )
    assert result.text == "hello world"


def test_empty_transcript():
    assert TranscriptResult(language=None).text == ""
