from video_ranker.tasks import app, process_chain


def names(c):
    return [t.task.rsplit(".", 1)[-1] for t in c.tasks]


def test_chain_from_url():
    c = process_chain(url="https://example.com/v")
    assert names(c) == ["ingest", "keyframes", "transcribe", "embed", "caption", "finalize"]
    assert c.tasks[0].args == ("https://example.com/v",)


def test_chain_from_id():
    c = process_chain(video_id=7)
    assert names(c)[0] == "keyframes"
    assert c.tasks[0].args == (7,)


def test_gpu_routing():
    routes = app.conf.task_routes
    assert routes["video_ranker.tasks.embed"]["queue"] == "gpu"
    assert "video_ranker.tasks.keyframes" not in routes
