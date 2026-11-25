from functools import lru_cache
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from video_ranker.config import get_settings


@lru_cache
def s3_client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint,
        aws_access_key_id=s.s3_access_key,
        aws_secret_access_key=s.s3_secret_key,
        region_name="us-east-1",
    )


def ensure_bucket(bucket: str | None = None) -> None:
    bucket = bucket or get_settings().s3_bucket
    client = s3_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_file(path: Path, key: str, bucket: str | None = None) -> str:
    s3_client().upload_file(str(path), bucket or get_settings().s3_bucket, key)
    return key


def upload_bytes(data: bytes, key: str, bucket: str | None = None) -> str:
    s3_client().put_object(Bucket=bucket or get_settings().s3_bucket, Key=key, Body=data)
    return key


def download_bytes(key: str, bucket: str | None = None) -> bytes:
    obj = s3_client().get_object(Bucket=bucket or get_settings().s3_bucket, Key=key)
    return obj["Body"].read()


def download_file(key: str, path: Path, bucket: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(bucket or get_settings().s3_bucket, key, str(path))
    return path
