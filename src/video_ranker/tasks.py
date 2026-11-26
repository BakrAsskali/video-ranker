from celery import Celery, Task, chain

from video_ranker.config import get_settings
from video_ranker.db.models import Video, VideoStatus
from video_ranker.db.session import SessionLocal
from video_ranker.ingest import pipeline

settings = get_settings()

app = Celery("video_ranker", broker=settings.redis_url, backend=settings.redis_url)
app.conf.update(
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    result_expires=24 * 3600,
    # models live on the gpu, keep those steps on their own worker
    task_routes={
        "video_ranker.tasks.transcribe": {"queue": "gpu"},
        "video_ranker.tasks.embed": {"queue": "gpu"},
        "video_ranker.tasks.caption": {"queue": "gpu"},
    },
)


class StepTask(Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        if not args or not isinstance(args[0], int):
            return
        with SessionLocal() as session:
            video = session.get(Video, args[0])
            if video is not None:
                video.status = VideoStatus.FAILED
                video.error = f"{self.name.rsplit('.', 1)[-1]}: {exc}"[:2000]
                session.commit()


@app.task(base=StepTask)
def ingest(url):
    return pipeline.ingest_url(url)


@app.task(base=StepTask)
def keyframes(video_id):
    with SessionLocal() as session:
        done = bool(session.get(Video, video_id).keyframes)
    if not done:
        pipeline.extract_video_keyframes(video_id)
    return video_id


@app.task(base=StepTask)
def transcribe(video_id):
    with SessionLocal() as session:
        done = session.get(Video, video_id).transcript is not None
    if not done:
        pipeline.transcribe_video(video_id)
    return video_id


@app.task(base=StepTask)
def embed(video_id):
    pipeline.embed_video(video_id)
    return video_id


@app.task(base=StepTask)
def caption(video_id):
    with SessionLocal() as session:
        done = any(f.caption for f in session.get(Video, video_id).keyframes)
    if not done:
        pipeline.caption_video(video_id)
    return video_id


@app.task(base=StepTask)
def finalize(video_id):
    with SessionLocal() as session:
        video = session.get(Video, video_id)
        video.status = VideoStatus.PROCESSED
        video.error = None
        session.commit()
    pipeline.write_record(video_id)
    return video_id


def process_chain(video_id=None, url=None):
    steps = [keyframes.s(), transcribe.s(), embed.s(), caption.s(), finalize.s()]
    if url is not None:
        return chain(ingest.s(url), *steps)
    steps[0] = keyframes.s(video_id)
    return chain(*steps)
