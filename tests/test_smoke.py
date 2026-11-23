from video_ranker.config import Settings
from video_ranker.db.models import Base


def test_database_url_uses_psycopg():
    s = Settings(postgres_password="pw", postgres_host="db", _env_file=None)
    assert s.database_url == "postgresql+psycopg://ranker:pw@db:5432/video_ranker"


def test_schema_tables():
    assert {"videos", "keyframes", "transcripts"} <= set(Base.metadata.tables)
