from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    postgres_user: str = "ranker"
    postgres_password: str = ""
    postgres_db: str = "video_ranker"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: str = ""
    s3_bucket: str = "videos"

    data_dir: str = "data"
    embedding_dim: int = 768

    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute: str = "auto"
    hf_token: str = ""

    clip_model: str = "hf-hub:timm/ViT-B-16-SigLIP"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
