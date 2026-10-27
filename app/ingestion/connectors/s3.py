"""Amazon S3 (or any S3-compatible store such as MinIO via ``endpoint_url``)."""

from __future__ import annotations

from typing import Any, Iterable

from app.ingestion.loaders import SUPPORTED_EXTENSIONS

from .base import RemoteDocument, secret_from_env


class S3Connector:
    """Config: ``bucket``, optional ``prefix``, ``region``, ``endpoint_url``. Credentials come from the
    standard AWS chain (IAM role, AWS_* env vars) or from env vars named by ``access_key_env`` /
    ``secret_key_env``."""

    def __init__(self, config: dict[str, Any], client: Any = None):
        import boto3

        self.name = config["name"]
        self.bucket = config["bucket"]
        self.prefix = config.get("prefix", "")
        if client is None:
            kwargs: dict[str, Any] = {}
            if config.get("region"):
                kwargs["region_name"] = config["region"]
            if config.get("endpoint_url"):
                kwargs["endpoint_url"] = config["endpoint_url"]
            if config.get("access_key_env"):
                kwargs["aws_access_key_id"] = secret_from_env(config, "access_key_env")
                kwargs["aws_secret_access_key"] = secret_from_env(config, "secret_key_env")
            client = boto3.client("s3", **kwargs)
        self.client = client

    def list_documents(self) -> Iterable[RemoteDocument]:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not any(key.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    continue
                yield RemoteDocument(
                    path=key[len(self.prefix):].lstrip("/"),
                    version=obj["ETag"].strip('"'),
                    fetch=lambda key=key: self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read(),
                )
